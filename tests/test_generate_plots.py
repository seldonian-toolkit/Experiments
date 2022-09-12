import os
import numpy as np
import pandas as pd

import pytest

from experiments.generate_plots import (
	SupervisedPlotGenerator,RLPlotGenerator)

def MSE(y_pred,y,**kwargs):
	n = len(y)
	res = sum(pow(y_pred-y,2))/n
	return res


def generate_episodes_and_calc_J(**kwargs):
    """ Calculate the expected discounted return 
    by generating episodes

    :return: episodes, J, where episodes is the list
        of generated ground truth episodes and J is
        the expected discounted return
    :rtype: (List(Episode),float)
    """
    # Get trained model weights from running the Seldonian algo
    model = kwargs['model']
    new_params = model.policy.get_params()
   
    # create env and agent
    hyperparameter_and_setting_dict = kwargs['hyperparameter_and_setting_dict']
    agent = create_agent(hyperparameter_and_setting_dict)
    env = create_env(hyperparameter_and_setting_dict)
   
    # set agent's weights to the trained model weights
    agent.set_new_params(new_params)
    
    # generate episodes
    num_episodes = kwargs['n_episodes_for_eval']
    episodes = run_trial_given_agent_and_env(
        agent=agent,env=env,num_episodes=num_episodes)

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,env.gamma) for ep in episodes])
    J = np.mean(returns)
    return episodes,J
	

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_supervised_plot_generator(gpa_regression_spec,experiment):
	np.random.seed(42)
	constraint_strs = ['Mean_Squared_Error - 3.0','2.0 - Mean_Squared_Error']
	deltas = [0.05,0.1]
	spec = gpa_regression_spec(constraint_strs,deltas)
	n_trials = 5
	# data_fracs = [0.001,0.05,0.5,1.0]
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

	spg.run_seldonian_experiment(verbose=True)

	## Make sure results file was created
	results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
	assert os.path.exists(results_file)

	# Make sure length of df is correct
	df = pd.read_csv(results_file)
	print(df)
	assert len(df) == 10
	dps = df.data_frac
	trial_is = df.trial_i
	perfs = df.performance
	passed_safetys = df.passed_safety
	faileds = df.failed
	
	assert dps[0] == 0.01
	assert trial_is[0] == 0

	assert dps[4] == 0.01
	assert trial_is[4] == 4

	assert dps[5] == 0.02
	assert trial_is[5] == 0

	assert dps[9] == 0.02
	assert trial_is[9] == 4
	
	# Make sure number of trial files created is correct
	trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
	trial_files = os.listdir(trial_dir)
	assert len(trial_files) == 10

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
	
