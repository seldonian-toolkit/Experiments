import os
import shutil

import pytest

from seldonian.models.models import LinearRegressionModel
from seldonian.RL.RL_model import RL_model
from seldonian.RL.Agents.Policies.Softmax import DiscreteSoftmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.models import objectives
from seldonian.utils.io_utils import (load_json,
	load_pickle,load_supervised_metadata)
from seldonian.dataset import DataSetLoader,RLDataSet
from seldonian.parse_tree.parse_tree import (ParseTree,
	make_parse_trees_from_constraints)
from seldonian.spec import createSupervisedSpec,createRLSpec

@pytest.fixture
def gpa_regression_spec():
	print("Setup gpa_regression_spec")
	
	def spec_maker(constraint_strs,deltas):

		data_pth = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'
		metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'

		(regime, sub_regime, columns,
	        sensitive_columns) = load_supervised_metadata(metadata_pth)
					
		include_sensitive_columns = False
		include_intercept_term = True

		# Load dataset from file
		loader = DataSetLoader(
			regime=regime)

		dataset = loader.load_supervised_dataset(
			filename=data_pth,
			metadata_filename=metadata_pth,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term,
			file_type='csv')

		spec = createSupervisedSpec(
			dataset=dataset,
			metadata_pth=metadata_pth,
			constraint_strs=constraint_strs,
			deltas=deltas,
			save=False,
			verbose=False)

		spec.optimization_hyperparams = {
				'lambda_init'   : 0.5,
				'alpha_theta'   : 0.005,
				'alpha_lamb'    : 0.005,
				'beta_velocity' : 0.9,
				'beta_rmsprop'  : 0.95,
				'num_iters'     : 50,
				'gradient_library': "autograd",
				'hyper_search'  : None,
				'verbose'       : True,
			}
		return spec
	
	yield spec_maker
	print("Teardown gpa_regression_spec")
	

@pytest.fixture
def gpa_classification_spec():
	print("Setup gpa_classification_spec")
	
	def spec_maker(constraint_strs,deltas):

		data_pth = 'static/datasets/supervised/GPA/gpa_classification_dataset.csv'
		metadata_pth = 'static/datasets/supervised/GPA/metadata_classification.json'

		(regime, sub_regime, columns,
	        sensitive_columns) = load_supervised_metadata(metadata_pth)
					
		include_sensitive_columns = False
		include_intercept_term = True

		# Load dataset from file
		loader = DataSetLoader(
			regime=regime)

		dataset = loader.load_supervised_dataset(
			filename=data_pth,
			metadata_filename=metadata_pth,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term,
			file_type='csv')

		spec = createSupervisedSpec(
			dataset=dataset,
			metadata_pth=metadata_pth,
			constraint_strs=constraint_strs,
			deltas=deltas,
			save=False,
			verbose=False)

		spec.optimization_hyperparams = {
				'lambda_init'   : 0.5,
				'alpha_theta'   : 0.005,
				'alpha_lamb'    : 0.005,
				'beta_velocity' : 0.9,
				'beta_rmsprop'  : 0.95,
				'num_iters'     : 50,
				'gradient_library': "autograd",
				'hyper_search'  : None,
				'verbose'       : True,
			}
		return spec
	
	yield spec_maker
	print("Teardown gpa_classification_spec")
	

@pytest.fixture
def gridworld_spec():
	print("Setup gridworld_spec")
	
	def spec_maker(constraint_strs,deltas):

		episodes_file = 'static/datasets/RL/gridworld/gridworld_1000episodes.pkl'
		episodes = load_pickle(episodes_file)
		dataset = RLDataSet(episodes=episodes)

		# Initialize policy
		num_states = 9
		observation_space = Discrete_Space(0, num_states-1)
		action_space = Discrete_Space(0, 3)
		env_description =  Env_Description(observation_space, action_space)
		policy = DiscreteSoftmax(hyperparam_and_setting_dict={},
			env_description=env_description)
		env_kwargs={'gamma':0.9}
		save_dir = '.'

		spec = createRLSpec(
			dataset=dataset,
			policy=policy,
			constraint_strs=constraint_strs,
			deltas=deltas,
			env_kwargs=env_kwargs,
			save=False,
			verbose=True)

		return spec
	
	yield spec_maker
	print("Teardown gridworld_spec")
	

@pytest.fixture
def experiment(request):
	results_dir = request.param
	""" Fixture to create and then remove results_dir and any files it may contain"""
	print("Setup experiment fixture")
	os.makedirs(results_dir,exist_ok=True)
	yield
	print("Teardown experiment fixture")
	shutil.rmtree(results_dir)


