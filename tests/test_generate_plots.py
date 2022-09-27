import os
import numpy as np
import pandas as pd

import pytest

from sklearn.metrics import log_loss,accuracy_score

from experiments.generate_plots import (
	SupervisedPlotGenerator,RLPlotGenerator)

from experiments.utils import MSE,generate_episodes_and_calc_J

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_regression_plot_generator(gpa_regression_spec,experiment):
	np.random.seed(42)
	constraint_strs = ['Mean_Squared_Error - 3.0','2.0 - Mean_Squared_Error']
	deltas = [0.05,0.1]
	spec = gpa_regression_spec(constraint_strs,deltas)
	n_trials = 2
	data_fracs = [0.01,0.02]
	datagen_method="resample"
	perf_eval_fn = MSE
	results_dir = "./tests/static/results"
	n_workers = 1
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
	# of the performance evaluation function,
	# which in our case is accuracy
	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		}
	
	spg = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		datagen_method=datagen_method,
		perf_eval_fn=perf_eval_fn,
		results_dir=results_dir,
		n_workers=n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs=perf_eval_kwargs,
		constraint_eval_kwargs={})
	
	assert spg.n_trials == n_trials
	assert spg.regime == 'supervised_learning'

	# Seldonian experiment

	spg.run_seldonian_experiment(verbose=True)

	## Make sure results file was created
	results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
	assert os.path.exists(results_file)

	# Make sure length of df is correct
	df = pd.read_csv(results_file)
	assert len(df) == 4
	dps = df.data_frac
	trial_is = df.trial_i
	perfs = df.performance
	passed_safetys = df.passed_safety
	faileds = df.failed
	
	assert dps[0] == 0.01
	assert trial_is[0] == 0

	assert dps[1] == 0.01
	assert trial_is[1] == 1

	assert dps[2] == 0.02
	assert trial_is[2] == 0

	assert dps[3] == 0.02
	assert trial_is[3] == 1
	
	# Make sure number of trial files created is correct
	trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
	trial_files = os.listdir(trial_dir)
	assert len(trial_files) == 4

	# Make sure the trial files have the right format
	trial_file_0 = os.path.join(trial_dir,trial_files[0])
	df_trial0 = pd.read_csv(trial_file_0)
	assert len(df_trial0) == 1

	# Now make plot
	savename = os.path.join(results_dir,"test_gpa_regression_plot.png")
	spg.make_plots(fontsize=12,legend_fontsize=8,
		performance_label='MSE',
		savename=savename)
	# Make sure it was saved
	assert os.path.exists(savename)

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_classification_plot_generator(gpa_classification_spec,experiment):
	np.random.seed(42)
	constraint_strs = ['abs((PR | [M]) - (PR | [F])) <= 0.2']
	deltas = [0.05]
	spec = gpa_classification_spec(constraint_strs,deltas)
	n_trials = 2
	data_fracs = [0.01,0.02]
	datagen_method="resample"
	performance_metric = 'accuracy'
	def perf_eval_fn(y_pred,y,**kwargs):
		if performance_metric == 'log_loss':
			return log_loss(y,y_pred)
		elif performance_metric == 'accuracy':
			return accuracy_score(y,y_pred > 0.5)

	perf_eval_fn = perf_eval_fn
	results_dir = "./tests/static/results"
	n_workers = 1
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
	# of the performance evaluation function,
	# which in our case is accuracy
	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		}
	
	spg = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		datagen_method=datagen_method,
		perf_eval_fn=perf_eval_fn,
		results_dir=results_dir,
		n_workers=n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs=perf_eval_kwargs,
		constraint_eval_kwargs={})
	
	assert spg.n_trials == n_trials
	assert spg.regime == 'supervised_learning'

	# Run two baselines and test that files are created correctly
	for baseline_model_name in ['random_classifier','logistic_regression']:
		spg.run_baseline_experiment(
			model_name=baseline_model_name,verbose=True)

		## Make sure results file was created
		baseline_results_file = os.path.join(results_dir,
			f"{baseline_model_name}_results/{baseline_model_name}_results.csv")
		assert os.path.exists(baseline_results_file)

		# Make sure length of df is correct
		baseline_df = pd.read_csv(baseline_results_file)
		assert len(baseline_df) == 4
		dps = baseline_df.data_frac
		trial_is = baseline_df.trial_i
		perfs = baseline_df.performance
		faileds = baseline_df.failed
	
		assert dps[0] == 0.01
		assert trial_is[0] == 0

		assert dps[1] == 0.01
		assert trial_is[1] == 1

		assert dps[2] == 0.02
		assert trial_is[2] == 0

		assert dps[3] == 0.02
		assert trial_is[3] == 1
		
		# Make sure number of trial files created is correct
		baseline_trial_dir = os.path.join(results_dir,
			f"{baseline_model_name}_results/trial_data")
		trial_files = os.listdir(baseline_trial_dir)
		assert len(trial_files) == 4

		# Make sure the trial files have the right format
		trial_file_0 = os.path.join(baseline_trial_dir,trial_files[0])
		df_trial0 = pd.read_csv(trial_file_0)
		assert len(df_trial0) == 1

	# Fairlearn baseline
	fairlearn_constraint_name = 'demographic_parity'
	fairlearn_epsilon_eval = 0.2 # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_eval_method = 'two-groups' # the epsilon used to evaluate g, needs to be same as epsilon in our definition
	fairlearn_epsilons_constraint = [0.2] # the epsilons used in the fitting constraint
	fairlearn_sensitive_feature_names=['M']
	
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

	for fairlearn_epsilon_constraint in fairlearn_epsilons_constraint:
		baseline_model_name = f'fairlearn_eps{fairlearn_epsilon_constraint:.2f}'
		spg.run_fairlearn_experiment(
			verbose=True,
			fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
			fairlearn_constraint_name=fairlearn_constraint_name,
			fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
			fairlearn_epsilon_eval=fairlearn_epsilon_eval,
			fairlearn_eval_kwargs=fairlearn_eval_kwargs,
			)
		## Make sure results file was created
		baseline_results_file = os.path.join(results_dir,
			f"{baseline_model_name}_results/{baseline_model_name}_results.csv")
		assert os.path.exists(baseline_results_file)

		# Make sure length of df is correct
		baseline_df = pd.read_csv(baseline_results_file)
	
		assert len(baseline_df) == 4
		dps = baseline_df.data_frac
		trial_is = baseline_df.trial_i
		perfs = baseline_df.performance
		faileds = baseline_df.failed
	
		assert dps[0] == 0.01
		assert trial_is[0] == 0

		assert dps[1] == 0.01
		assert trial_is[1] == 1

		assert dps[2] == 0.02
		assert trial_is[2] == 0

		assert dps[3] == 0.02
		assert trial_is[3] == 1
		
		# Make sure number of trial files created is correct
		baseline_trial_dir = os.path.join(results_dir,
			f"{baseline_model_name}_results/trial_data")
		trial_files = os.listdir(baseline_trial_dir)
		assert len(trial_files) == 4

		# Make sure the trial files have the right format
		trial_file_0 = os.path.join(baseline_trial_dir,trial_files[0])
		df_trial0 = pd.read_csv(trial_file_0)
		assert len(df_trial0) == 1

	# Seldonian experiment

	spg.run_seldonian_experiment(verbose=True)

	## Make sure results file was created
	results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
	assert os.path.exists(results_file)

	# Make sure length of df is correct
	df = pd.read_csv(results_file)
	assert len(df) == 4
	dps = df.data_frac
	trial_is = df.trial_i
	perfs = df.performance
	passed_safetys = df.passed_safety
	faileds = df.failed
	
	assert dps[0] == 0.01
	assert trial_is[0] == 0

	assert dps[1] == 0.01
	assert trial_is[1] == 1

	assert dps[2] == 0.02
	assert trial_is[2] == 0

	assert dps[3] == 0.02
	assert trial_is[3] == 1
	
	# Make sure number of trial files created is correct
	trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
	trial_files = os.listdir(trial_dir)
	assert len(trial_files) == 4

	# Make sure the trial files have the right format
	trial_file_0 = os.path.join(trial_dir,trial_files[0])
	df_trial0 = pd.read_csv(trial_file_0)
	assert len(df_trial0) == 1

	# Now make plot
	savename = os.path.join(results_dir,"test_gpa_regression_plot.png")
	spg.make_plots(fontsize=12,legend_fontsize=8,
		performance_label='MSE',
		savename=savename)
	# Make sure it was saved
	assert os.path.exists(savename)

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_too_few_datapoints(gpa_regression_spec,experiment):
	""" Test that too small of a data_frac resulting in < 1
	data points in a trial raises an error """
	np.random.seed(42)
	constraint_strs = ['Mean_Squared_Error <= 2.0']
	deltas = [0.05]
	spec = gpa_regression_spec(constraint_strs,deltas)
	n_trials = 1
	data_fracs = [0.000001]
	datagen_method="resample"
	perf_eval_fn = MSE
	results_dir = "./tests/static/results"
	n_workers = 1
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
	# of the performance evaluation function,
	# which in our case is accuracy
	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		}
	
	spg = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		datagen_method=datagen_method,
		perf_eval_fn=perf_eval_fn,
		results_dir=results_dir,
		n_workers=n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs=perf_eval_kwargs,
		constraint_eval_kwargs={})
	
	assert spg.n_trials == n_trials
	assert spg.regime == 'supervised_learning'

	with pytest.raises(ValueError) as excinfo:
		spg.run_seldonian_experiment(verbose=True)
	error_str = (
		f"This data_frac={data_fracs[0]} "
		f"results in 0 data points. "
		 "Must have at least 1 data point to run a trial.")

	assert str(excinfo.value) == error_str

@pytest.mark.parametrize('experiment', ["./tests/static/gridworld_results"], indirect=True)
def test_too_few_episodes(gridworld_spec,experiment):
	""" Test that too small of a data_frac resulting in < 1
	episodes in a trial raises an error """
	np.random.seed(42)
	constraint_strs = ['J_pi_new >= -0.25']
	deltas=[0.05]
	spec = gridworld_spec(constraint_strs,deltas)
	n_trials = 1
	data_fracs = [0.000001]
	datagen_method="generate_episodes"
	perf_eval_fn = generate_episodes_and_calc_J
	results_dir = "./tests/static/gridworld_results"
	n_workers = 1
	# Get performance evaluation kwargs set up
	n_episodes_for_eval = 1000
	perf_eval_kwargs = {'n_episodes_for_eval':n_episodes_for_eval}
	
	hyperparameter_and_setting_dict = {}
	hyperparameter_and_setting_dict["env"] = "gridworld"
	hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
	hyperparameter_and_setting_dict["num_episodes"] = 1000
	hyperparameter_and_setting_dict["num_trials"] = 1
	hyperparameter_and_setting_dict["vis"] = False

	spg = RLPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		datagen_method=datagen_method,
		hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
		perf_eval_fn=perf_eval_fn,
		results_dir=results_dir,
		n_workers=n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs=perf_eval_kwargs,
		constraint_eval_kwargs={})
	
	assert spg.n_trials == n_trials
	assert spg.regime == 'reinforcement_learning'

	with pytest.raises(ValueError) as excinfo:
		spg.run_seldonian_experiment(verbose=True)
	error_str = (
		f"This data_frac={data_fracs[0]} "
		f"results in 0 episodes. "
		 "Must have at least 1 episode to run a trial.")

	assert str(excinfo.value) == error_str

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_RL_plot_generator(gridworld_spec,experiment):
	np.random.seed(42)
	constraint_strs = ['J_pi_new >= - 0.25']
	deltas = [0.05]
	spec = gridworld_spec(constraint_strs,deltas)
	spec.optimization_hyperparams['num_iters'] = 20
	n_trials = 2
	data_fracs = [0.01,0.1]
	datagen_method="generate_episodes"
	perf_eval_fn = generate_episodes_and_calc_J
	results_dir = "./tests/static/results"
	n_workers = 4
	n_episodes_for_eval=1000
	# Get performance evaluation kwargs set up
	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	
	# Define any additional keyword arguments (besides theta)
	# of the performance evaluation function,
	perf_eval_kwargs = {
		'n_episodes_for_eval':n_episodes_for_eval
	}
	
	hyperparameter_and_setting_dict = {}
	hyperparameter_and_setting_dict["env"] = "gridworld"
	hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
	hyperparameter_and_setting_dict["num_episodes"] = 1000
	hyperparameter_and_setting_dict["num_trials"] = 1
	hyperparameter_and_setting_dict["vis"] = False

	spg = RLPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		datagen_method=datagen_method,
		hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
		perf_eval_fn=perf_eval_fn,
		results_dir=results_dir,
		n_workers=n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs=perf_eval_kwargs,
		constraint_eval_kwargs={})
	
	assert spg.n_trials == n_trials
	assert spg.regime == 'reinforcement_learning'

	# Seldonian experiment

	spg.run_seldonian_experiment(verbose=True)

	## Make sure results file was created
	results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
	assert os.path.exists(results_file)

	# Make sure length of df is correct
	df = pd.read_csv(results_file)
	assert len(df) == 4
	dps = df.data_frac
	trial_is = df.trial_i
	perfs = df.performance
	passed_safetys = df.passed_safety
	faileds = df.failed
	
	assert dps[0] == 0.01
	assert trial_is[0] == 0

	assert dps[1] == 0.01
	assert trial_is[1] == 1

	assert dps[2] == 0.1
	assert trial_is[2] == 0

	assert dps[3] == 0.1
	assert trial_is[3] == 1
	
	# Make sure number of trial files created is correct
	trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
	trial_files = os.listdir(trial_dir)
	assert len(trial_files) == 4

	# Make sure the trial files have the right format
	trial_file_0 = os.path.join(trial_dir,trial_files[0])
	df_trial0 = pd.read_csv(trial_file_0)
	assert len(df_trial0) == 1

	# Now make plot
	savename = os.path.join(results_dir,"test_gridworld_plot.png")
	spg.make_plots(fontsize=12,legend_fontsize=8,
		performance_label='-IS_estimate',
		savename=savename)
	# Make sure it was saved
	assert os.path.exists(savename)
	
