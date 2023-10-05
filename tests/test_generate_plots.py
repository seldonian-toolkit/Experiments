import os
import numpy as np
import pandas as pd

import pytest

from experiments.generate_plots import (
	SupervisedPlotGenerator,RLPlotGenerator)

from experiments.experiment_utils import (
	generate_episodes_and_calc_J,has_failed)

from experiments.perf_eval_funcs import (MSE,probabilistic_accuracy)

from seldonian.RL.environments.gridworld import Gridworld

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_regression_plot_generator(gpa_regression_spec,experiment):
	np.random.seed(42)
	constraint_strs = ['Mean_Squared_Error - 3.0','2.0 - Mean_Squared_Error']
	deltas = [0.05,0.1]
	spec = gpa_regression_spec(constraint_strs,deltas)
	n_trials = 2
	data_fracs = [0.01,0.1]
	datagen_method="resample"
	perf_eval_fn = MSE
	results_dir = "./tests/static/results"
	n_workers = 1
	# Get performance evaluation kwargs set up
	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset

	test_features = dataset.features
	test_labels = dataset.labels

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
	gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
	
	assert dps[0] == 0.01
	assert trial_is[0] == 0
	assert passed_safetys[0] == False
	assert gvecs.str[0][0] == -np.inf

	assert dps[1] == 0.01
	assert trial_is[1] == 1
	assert passed_safetys[1] == False
	assert gvecs.str[0][1] == -np.inf

	assert dps[2] == 0.1
	assert trial_is[2] == 0
	assert passed_safetys[2] == False
	assert gvecs.str[0][2] == -np.inf

	assert dps[3] == 0.1
	assert trial_is[3] == 1
	assert passed_safetys[3] == False
	assert gvecs.str[0][3] == -np.inf
	
	# Make sure number of trial files created is correct
	trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
	trial_files = os.listdir(trial_dir)
	assert len(trial_files) == 4

	# Make sure the trial files have the right format
	trial_file_0 = os.path.join(trial_dir,trial_files[0])
	df_trial0 = pd.read_csv(trial_file_0)
	assert len(df_trial0) == 1

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

	test_features = dataset.features
	test_labels = dataset.labels

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
	constraint_strs = ['J_pi_new_IS >= -0.25']
	deltas=[0.05]
	spec = gridworld_spec(constraint_strs,deltas)
	n_trials = 1
	data_fracs = [0.000001]
	datagen_method="generate_episodes"
	perf_eval_fn = generate_episodes_and_calc_J
	results_dir = "./tests/static/gridworld_results"
	n_workers = 1
	# Get performance evaluation kwargs set up
	n_episodes_for_eval = 100
	perf_eval_kwargs = {'n_episodes_for_eval':n_episodes_for_eval}
	
	hyperparameter_and_setting_dict = {}
	hyperparameter_and_setting_dict["env"] = Gridworld()
	hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
	hyperparameter_and_setting_dict["num_episodes"] = 100
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
	constraint_strs = ['J_pi_new_IS >= - 0.25']
	deltas = [0.05]
	spec = gridworld_spec(constraint_strs,deltas)
	spec.optimization_hyperparams['num_iters'] = 10
	n_trials = 2
	data_fracs = [0.05,0.1]
	datagen_method="generate_episodes"
	perf_eval_fn = generate_episodes_and_calc_J
	results_dir = "./tests/static/results"
	n_workers = 1
	n_episodes_for_eval=100
	# Get performance evaluation kwargs set up
	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	
	# Define any additional keyword arguments (besides theta)
	# of the performance evaluation function,
	perf_eval_kwargs = {
		'n_episodes_for_eval':n_episodes_for_eval
	}
	
	hyperparameter_and_setting_dict = {}
	hyperparameter_and_setting_dict["env"] = Gridworld()
	hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
	hyperparameter_and_setting_dict["num_episodes"] = 100
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
	print("df:")
	print(df)
	
	assert dps[0] == 0.05
	assert trial_is[0] == 0

	assert dps[1] == 0.05
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
		save_format="png",
		savename=savename)
	# Make sure it was saved
	assert os.path.exists(savename)
