import pytest
import numpy as np

from experiments.baselines.baselines import (
	SupervisedExperimentBaseline,RLExperimentBaseline)
from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.environments.gridworld import Gridworld

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






	
