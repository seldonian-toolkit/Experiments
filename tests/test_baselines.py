import pytest
import numpy as np

from experiments.baselines.baselines import (
	SupervisedExperimentBaseline,RLExperimentBaseline)
from experiments.baselines.fitted_Q import (
	ExactTabularFittedQBaseline,ApproximateTabularFittedQBaseline)
from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.environments.gridworld import Gridworld
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def test_supervised_base_class():
	bl_model = SupervisedExperimentBaseline(model_name="custom_SL_baseline")
	assert bl_model.model_name == 'custom_SL_baseline'
	with pytest.raises(NotImplementedError) as excinfo:
		bl_model.train(np.random.randn(2,4),np.random.randn(2))
	error_str = ("Implement this method in a child class")

	assert str(excinfo.value) == error_str

def test_RL_base_class():
	gw = Gridworld(size=3)
	env_description = gw.get_env_description()
	policy = DiscreteSoftmax(hyperparam_and_setting_dict={},env_description=env_description)
	bl_model = RLExperimentBaseline(model_name="custom_RL_baseline",policy=policy)
	assert bl_model.model_name == 'custom_RL_baseline'
	assert bl_model.gamma == 1.0

	with pytest.raises(NotImplementedError) as excinfo:
		bl_model.train(None)
	error_str = ("Implement this method in a child class")

	assert str(excinfo.value) == error_str


def test_fitted_Q_baseline(gridworld_spec):
	""" Set up and train a fitted Q on gridworld
	using a uniform random softmax behavior policy
	"""
	constraint_strs = ['J_pi_new_IS >= -0.25']
	deltas=[0.05]
	spec = gridworld_spec(constraint_strs,deltas)



	# First exact tabular
	gw = Gridworld(size=3)
	env_description = gw.get_env_description()
	policy = DiscreteSoftmax(hyperparam_and_setting_dict={},env_description=env_description)
	num_observations=9
	num_actions=4
	bl_model1 = ExactTabularFittedQBaseline(
		model_name="Tabular_fitted_Q",
		regressor_class=LinearRegression,
		policy=policy,
		env_kwargs={
			'gamma':0.9,
			'num_observations':num_observations,
			'num_actions':num_actions,
			'terminal_observation':8
		},
		num_iters=10
	)
	assert bl_model1.model_name == 'Tabular_fitted_Q'
	assert bl_model1.gamma == 0.9
	bl_model1.policy.set_new_params(np.ones((num_observations,num_actions)))
	assert np.allclose(bl_model1.policy.get_params(),np.ones((num_observations,num_actions)))
	bl_model1.reset_policy_params()
	assert np.allclose(bl_model1.policy.get_params(),np.zeros((num_observations,num_actions)))

	fitted_params1 = bl_model1.train(spec.dataset)
	# Test if we got optimal policy by checking greedy actions
	# 0,1,2,3 are up,right,down,left
	fitted_greedy_actions1 = np.array([np.argmax(bl_model1.policy.get_params()[obs]) for obs in range(bl_model1.num_observations)])
	assert fitted_greedy_actions1[0] in [1,2]
	assert fitted_greedy_actions1[1] in [1,2]
	assert fitted_greedy_actions1[2] == 2
	assert fitted_greedy_actions1[3] == 1
	assert fitted_greedy_actions1[4] == 1
	assert fitted_greedy_actions1[5] == 2
	assert fitted_greedy_actions1[6] == 0
	assert fitted_greedy_actions1[7] == 1
	assert fitted_greedy_actions1[8] in [0,1,2,3]
	
	# Now approximate tabular
	bl_model2 = ApproximateTabularFittedQBaseline(
		model_name="Approx_tabular_fitted_Q",
		regressor_class=RandomForestRegressor,
		policy=policy,
		env_kwargs={
			'gamma':0.9,
			'num_observations':num_observations,
			'num_actions':num_actions,
			'terminal_observation':8
		},
		num_iters=10
	)

	assert bl_model2.model_name == 'Approx_tabular_fitted_Q'
	assert bl_model2.gamma == 0.9
	bl_model2.policy.set_new_params(np.ones((num_observations,num_actions)))
	assert np.allclose(bl_model2.policy.get_params(),np.ones((num_observations,num_actions)))
	bl_model2.reset_policy_params()
	assert np.allclose(bl_model2.policy.get_params(),np.zeros((num_observations,num_actions)))

	fitted_params2 = bl_model2.train(spec.dataset)
	fitted_greedy_actions2 = np.array([np.argmax(bl_model2.policy.get_params()[obs]) for obs in range(bl_model2.num_observations)])
	assert fitted_greedy_actions2[0] in [1,2]
	assert fitted_greedy_actions2[1] in [1,2]
	assert fitted_greedy_actions2[2] == 2
	assert fitted_greedy_actions2[3] == 1
	assert fitted_greedy_actions2[4] == 1
	assert fitted_greedy_actions2[5] == 2
	assert fitted_greedy_actions2[6] == 0
	assert fitted_greedy_actions2[7] == 1
	assert fitted_greedy_actions2[8] in [0,1,2,3]
	





	
