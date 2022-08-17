import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score


if __name__ == "__main__":
	# Parameter setup
	run_experiments = True
	make_plots = True
	save_plot = False
	constraint_name = 'predictive_equality'
	fairlearn_constraint_name = constraint_name
	fairlearn_epsilon_eval = 0.2 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_eval_method = 'two-groups' # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_epsilons_constraint = [0.01,0.1,1.0] # the epsilons used in the fitting constraint
	performance_metric = 'accuracy'
	n_trials = 25
	data_fracs = np.logspace(-4,0,15)
	n_workers = 8
	results_dir = f'results/gpa_{constraint_name}_{performance_metric}'
	plot_savename = os.path.join(results_dir,f'{constraint_name}_{performance_metric}.png')

	verbose=True

	# Load spec
	specfile = f'../interface_outputs/gpa_{constraint_name}/spec.pkl'
	spec = load_pickle(specfile)

	spec.primary_objective = spec.model_class().sample_logistic_loss
	spec.use_builtin_primary_gradient_fn = False
	spec.optimization_hyperparams['alpha_theta'] = 0.01
	spec.optimization_hyperparams['alpha_lamb'] = 0.01
	spec.optimization_hyperparams['num_iters'] = 1000

	os.makedirs(results_dir,exist_ok=True)

	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	label_column = dataset.label_column
	include_sensitive_columns = dataset.include_sensitive_columns
	include_intercept_term = dataset.include_intercept_term

	test_features = dataset.df.loc[:,
		dataset.df.columns != label_column]
	test_labels = dataset.df[label_column]

	if not include_sensitive_columns:
		test_features = test_features.drop(
			columns=dataset.sensitive_column_names) 

	if include_intercept_term:
		test_features.insert(0,'offset',1.0) # inserts a column of 1's in place

	# Setup performance evaluation function and kwargs 
	# of the performance evaluation function

	# perf_eval_fn = lambda y_pred,y,X: fbeta_score(y,y_pred,beta=2)
	def perf_eval_fn(y_pred,y,**kwargs):
		if performance_metric == 'log_loss':
			return log_loss(y,y_pred)
		elif performance_metric == 'accuracy':
			return accuracy_score(y,y_pred > 0.5)

	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		}

	plot_generator = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		n_workers=n_workers,
		datagen_method='resample',
		perf_eval_fn=perf_eval_fn,
		constraint_eval_fns=[],
		results_dir=results_dir,
		perf_eval_kwargs=perf_eval_kwargs,
		)

	# # Baseline models
	if run_experiments:
		plot_generator.run_baseline_experiment(
			model_name='random_classifier',verbose=True)

		plot_generator.run_baseline_experiment(
			model_name='logistic_regression',verbose=True)

		# Seldonian experiment
		plot_generator.run_seldonian_experiment(verbose=verbose)


	######################
	# Fairlearn experiment 
	######################

	fairlearn_sensitive_feature_names=['M']
	
	# Make dict of test set features, labels and sensitive feature vectors
	
	# Make dict of test set features, labels and sensitive feature vectors
	if 'offset' in test_features.columns:
		test_features_fairlearn = test_features.drop(columns=['offset'])
	else:
		test_features_fairlearn = test_features
	fairlearn_eval_kwargs = {
		'X':test_features_fairlearn,
		'y':test_labels,
		'sensitive_features':dataset.df.loc[:,
			fairlearn_sensitive_feature_names],
		'eval_method':fairlearn_eval_method,
		}

	if run_experiments:
		for fairlearn_epsilon_constraint in fairlearn_epsilons_constraint:
			plot_generator.run_fairlearn_experiment(
				verbose=verbose,
				fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
				fairlearn_constraint_name=fairlearn_constraint_name,
				fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
				fairlearn_epsilon_eval=fairlearn_epsilon_eval,
				fairlearn_eval_kwargs=fairlearn_eval_kwargs,
				)

	if make_plots:
		if save_plot:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric,
				savename=plot_savename)
		else:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric)