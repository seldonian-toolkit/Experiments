import os
import numpy as np 
import tracemalloc,linecache

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score

def perf_eval_fn(y_pred,y,**kwargs):
	performance_metric = kwargs['performance_metric']
	if performance_metric == 'Accuracy':
		v = np.where(y!=1.0,1.0-y_pred,y_pred)
		return sum(v)/len(v)
		

def main():
	# Parameter setup
	run_experiments = True
	make_plots = True
	save_plot = False
	include_legend = True
	constraint_name = 'disparate_impact'
	fairlearn_constraint_name = constraint_name
	fairlearn_epsilon_eval = 0.8 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_eval_method = 'two-groups' # two-groups is the Seldonian definition, 'native' is the fairlearn definition
	# fairlearn_epsilons_constraint = [0.01,0.1,1.0] # the epsilons used in the fitting constraint
	fairlearn_epsilons_constraint = [0.01] # the epsilons used in the fitting constraint
	performance_metric = 'Accuracy'
	n_trials = 10
	data_fracs = np.logspace(-4,0,15)
	n_workers = 7
	results_dir = f'../../results/gpa/{constraint_name}_{performance_metric}_2023Feb10'
	plot_savename = os.path.join(results_dir,f'gpa_{constraint_name}_{performance_metric}.png')

	verbose=True

	# Load spec
	specfile = f'../../../engine-repo-dev/examples/GPA_tutorial/{constraint_name}/spec.pkl'
	spec = load_pickle(specfile)

	os.makedirs(results_dir,exist_ok=True)

	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	test_features = dataset.features
	test_labels = dataset.labels

	# Setup performance evaluation function and kwargs 

	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		'performance_metric':performance_metric
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
		'performance_metric':performance_metric,
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
		plot_generator.make_plots(fontsize=12,legend_fontsize=8,
			performance_label=performance_metric,
			include_legend=include_legend,
			savename=plot_savename if save_plot else None)



if __name__ == "__main__":

	main()