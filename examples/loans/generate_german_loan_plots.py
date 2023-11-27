import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score

from experiments.baselines.logistic_regression import BinaryLogisticRegressionBaseline
from experiments.baselines.random_classifiers import (
    UniformRandomClassifierBaseline)
from experiments.baselines.random_forest import RandomForestClassifierBaseline

def perf_eval_fn(y_pred,y,**kwargs):
	return log_loss(y,y_pred)

if __name__ == "__main__":
	# Parameter setup
	run_experiments = True
	make_plots = True
	save_plot = True

	model_label_dict = {
		'qsa':'Seldonian model',
		'random_classifier': 'Uniform random',
		'logistic_regression': 'Logistic regression (no constraint)',
		'random_forest': 'Random forest (no constraint)',
		'fairlearn_eps0.01': 'Fairlearn (model 4)',
		'fairlearn_eps0.10': 'Fairlearn (model 3)',
		'fairlearn_eps0.90': 'Fairlearn (model 2)',
		'fairlearn_eps1.00': 'Fairlearn (model 1)',
		}

	constraint_name = 'disparate_impact'
	fairlearn_constraint_name = constraint_name
	fairlearn_epsilon_eval = 0.9 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_eval_method = 'two-groups' # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_epsilons_constraint = [0.01,0.1,0.9,1.0] # the epsilons used in the fitting constraint
	performance_metric = 'Log loss'
	n_trials = 5
	data_fracs = np.logspace(-3,0,15)
	n_workers = 6
	verbose=False
	results_dir = f'../../results/loan_{constraint_name}_debug_2023Jul20'
	os.makedirs(results_dir,exist_ok=True)

	plot_savename = os.path.join(results_dir,f'{constraint_name}_{performance_metric}.png')

	# Load spec
	specfile = f'./data/spec/loans_{constraint_name}_{fairlearn_epsilon_eval}_spec.pkl'
	spec = load_pickle(specfile)

	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	test_features = dataset.features
	test_labels = dataset.labels

	# Setup performance evaluation function and kwargs 
	# of the performance evaluation function

	

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
			baseline_model=RandomForestClassifierBaseline(),verbose=verbose)

		plot_generator.run_baseline_experiment(
			baseline_model=BinaryLogisticRegressionBaseline(),verbose=verbose)

		# Seldonian experiment
		plot_generator.run_seldonian_experiment(verbose=verbose)

	# ######################
	# # Fairlearn experiment 
	# ######################
	# fairlearn_sensitive_feature_names = ['M']
	# fairlearn_sensitive_col_indices = [dataset.sensitive_col_names.index(
	#     col) for col in fairlearn_sensitive_feature_names]
	# fairlearn_sensitive_features = dataset.sensitive_attrs[:,fairlearn_sensitive_col_indices]
	# # Setup ground truth test dataset for Fairlearn
	# test_features_fairlearn = test_features
	# fairlearn_eval_kwargs = {
	# 	'X':test_features_fairlearn,
	# 	'y':test_labels,
	# 	'sensitive_features':fairlearn_sensitive_features,
	# 	'eval_method':fairlearn_eval_method,
	# 	}

	# if run_experiments:
	# 	for fairlearn_epsilon_constraint in fairlearn_epsilons_constraint:
	# 		plot_generator.run_fairlearn_experiment(
	# 			verbose=verbose,
	# 			fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
	# 			fairlearn_constraint_name=fairlearn_constraint_name,
	# 			fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
	# 			fairlearn_epsilon_eval=fairlearn_epsilon_eval,
	# 			fairlearn_eval_kwargs=fairlearn_eval_kwargs,
	# 			)

	if make_plots:
		plot_generator.make_plots(fontsize=12,legend_fontsize=8,
			performance_label=performance_metric,
			performance_yscale='log',
			model_label_dict=model_label_dict,
			savename=plot_savename if save_plot else None,
			save_format="png")
