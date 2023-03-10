import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score


if __name__ == "__main__":
	# Parameter setup
	run_experiments = False
	make_plots = True
	save_plot = False
	constraint_name = 'disparate_impact'
	fairlearn_constraint_name = constraint_name
	fairlearn_epsilon_eval = 0.9 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_eval_method = 'native' 
	fairlearn_epsilons_constraint = [0.9] # the epsilons used in the fitting constraint
	performance_metric = 'log_loss'
	n_trials = 50
	data_fracs = np.logspace(-3,0,15)
	n_workers = 8
	verbose=True
	results_dir = f'../../results/loan_{constraint_name}_fairlearndef_log_loss_2023Jan09'
	os.makedirs(results_dir,exist_ok=True)

	plot_savename = os.path.join(results_dir,f'{constraint_name}_{performance_metric}.png')

	# Load spec
	specfile = f'../../../engine-repo-dev/examples/loan_tutorial/spec_fairlearn.pkl'
	spec = load_pickle(specfile)

	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	test_features = dataset.features
	test_labels = dataset.labels

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

	if run_experiments:
		# Baseline models
		plot_generator.run_baseline_experiment(
			model_name='random_classifier',verbose=verbose)

		plot_generator.run_baseline_experiment(
			model_name='logistic_regression',verbose=verbose)

		# Seldonian experiment
		plot_generator.run_seldonian_experiment(verbose=verbose)

	######################
	# Fairlearn experiment 
	######################
	fairlearn_sensitive_feature_names = ['M']
	fairlearn_sensitive_col_indices = [dataset.sensitive_col_names.index(
	    col) for col in fairlearn_sensitive_feature_names]
	fairlearn_sensitive_features = dataset.sensitive_attrs[:,fairlearn_sensitive_col_indices]
	# Setup ground truth test dataset for Fairlearn
	test_features_fairlearn = test_features
	fairlearn_eval_kwargs = {
		'X':test_features_fairlearn,
		'y':test_labels,
		'sensitive_features':fairlearn_sensitive_features,
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
				performance_yscale='log',
				savename=plot_savename)
		else:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric,
				performance_yscale='log')