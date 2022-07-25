import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.utils import generate_resampled_datasets
from experiments.generate_plots import SupervisedPlotGenerator

from seldonian.utils.io_utils import load_pickle

from sklearn.metrics import fbeta_score,log_loss

def weighted_loss(y_pred,y,**kwargs):
	""" Calculate the averaged weighted cost: 
	sum_i p_(wrong answer for point I) * c_i
	where c_i is 1 for false positives and 5 for false negatives

	:param model: The Seldonian model object 
	:type model: :py:class:`.SeldonianModel` object

	:param solution: The parameter weights
	:type solution: numpy ndarray

	:param X: The features
	:type X: numpy ndarray

	:param Y: The labels
	:type Y: numpy ndarray

	:return: weighted loss such that false negatives 
		have 5 times the cost as false positives
	:rtype: float
	"""
	# calculate probabilistic false positive rate and false negative rate
	n_points = len(y)
	neg_mask = y!=1 # this includes false positives and true negatives
	pos_mask = y==1 # this includes true positives and false negatives
	fp_values = y_pred[neg_mask] # get just false positives
	fn_values = 1.0-y_pred[pos_mask] # get just false negatives
	fpr = 1.0*np.sum(fp_values)
	fnr = 5.0*np.sum(fn_values)
	return (fpr + fnr)/n_points



if __name__ == "__main__":
	# Load spec
	interface_output_dir = os.path.join('/Users/ahoag/beri/code',
		'interface_outputs/loan_predictive_equality')
	specfile = os.path.join(interface_output_dir,'spec.pkl')
	spec = load_pickle(specfile)
	spec.primary_objective = spec.model_class().sample_logistic_loss
	spec.use_builtin_primary_gradient_fn = False
	spec.optimization_hyperparams['alpha_theta'] = 0.01
	spec.optimization_hyperparams['alpha_lamb'] = 0.01
	spec.optimization_hyperparams['num_iters'] = 1000

	### PARAMETER SETUP ###
	n_trials = 25
	data_pcts = np.logspace(-1.3,0,15)
	# data_pcts = [0.005]
	n_workers=8
	results_dir = './results/loan_predictive_equality_log_loss'
	# results_dir = './results/loan_predictive_equality'
	verbose=True

	os.makedirs(results_dir,exist_ok=True)

	# Get performance evaluation kwargs set up
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

	# Define any additional keyword arguments (besides theta)
	# of the performance evaluation function

	# perf_eval_fn = lambda y_pred,y,X: fbeta_score(y,y_pred,beta=2)
	def perf_eval_fn(y_pred,y,**kwargs):
		return log_loss(y,y_pred)

	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		}

	plot_generator = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_pcts=data_pcts,
		n_workers=n_workers,
		datagen_method='resample',
		perf_eval_fn=perf_eval_fn,
		constraint_eval_fns=[],
		results_dir=results_dir,
		perf_eval_kwargs=perf_eval_kwargs,
		)

	# # # Baseline models
	# plot_generator.run_baseline_experiment(
	# 	model_name='dummy_classifier',verbose=True)

	# plot_generator.run_baseline_experiment(
	# 	model_name='logistic_regression',verbose=True)

	# # Seldonian experiment
	# plot_generator.run_seldonian_experiment(verbose=verbose)

	######################
	# Fairlearn experiment 
	######################

	# The way Fairlearn handles sensitive columns is different than 
	# Seldonian way of handling it. In Seldonian, we one-hot encode
	# sensitive columns. In Fairlearn, they integer encode. 
	# So we have two columns, M and F, 
	# whereas they would have one: "sex". It turns out that our 
	# M column encodes both sexes since it is binary, so we 
	# can just tell Fairlearn to use that column.
	fairlearn_sensitive_feature_names=['M']
	
	fairlearn_constraint_name = "predictive_equality"
	fairlearn_epsilon_constraint = 0.2 # the epsilon used in the fitting constraint
	fairlearn_epsilon_eval = 0.2 # the epsilon used in the evaluation constraint

	# Make dict of test set features, labels and sensitive feature vectors
	
	fairlearn_eval_kwargs = {
		'X':test_features.drop(columns=['offset']),
		'y':test_labels,
		'sensitive_features':dataset.df.loc[:,
			fairlearn_sensitive_feature_names]
		}

	# plot_generator.run_fairlearn_experiment(
	# 	verbose=verbose,
	# 	fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
	# 	fairlearn_constraint_name=fairlearn_constraint_name,
	# 	fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
	# 	fairlearn_epsilon_eval=fairlearn_epsilon_eval,
	# 	fairlearn_eval_kwargs=fairlearn_eval_kwargs,
	# 	)

	savename = os.path.join(results_dir,'predictive_equality_log_loss.png')
	
	plot_generator.make_plots(fontsize=12,legend_fontsize=8,
		performance_label='log loss',best_performance=None)

	# plot_generator.make_plots(fontsize=12,legend_fontsize=8,
	# 	performance_label='log loss',best_performance=None,
	# 	savename=savename)


