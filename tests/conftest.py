import os
import shutil

import pytest

from seldonian.models.models import LinearRegressionModel
from seldonian.RL.RL_model import RL_model
from seldonian.RL.Agents.Policies.Softmax import Softmax
from seldonian.RL.Env_Description.Env_Description import Env_Description
from seldonian.RL.Env_Description.Spaces import Discrete_Space
from seldonian.models import objectives
from seldonian.utils.io_utils import load_json,load_pickle
from seldonian.dataset import DataSetLoader,RLDataSet
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.spec import SupervisedSpec,createRLSpec

@pytest.fixture
def gpa_regression_spec():
	print("Setup gpa_regression_spec")
	
	def spec_maker(constraint_strs,deltas):

		data_pth = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'
		metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'

		metadata_dict = load_json(metadata_pth)
		regime = metadata_dict['regime']
		print(f"Regime={regime}")
		sub_regime = metadata_dict['sub_regime']
		columns = metadata_dict['columns']
		sensitive_columns = metadata_dict['sensitive_columns']
					
		include_sensitive_columns = False
		include_intercept_term = True

		model = LinearRegressionModel()

		# Mean squared error
		primary_objective = objectives.Mean_Squared_Error

		# Load dataset from file
		loader = DataSetLoader(
			regime=regime)

		dataset = loader.load_supervised_dataset(
			filename=data_pth,
			metadata_filename=metadata_pth,
			include_sensitive_columns=include_sensitive_columns,
			include_intercept_term=include_intercept_term,
			file_type='csv')

		# For each constraint, make a parse tree
		parse_trees = []
		for ii in range(len(constraint_strs)):
			constraint_str = constraint_strs[ii]

			delta = deltas[ii]
			# Create parse tree object
			parse_tree = ParseTree(delta=delta,
				regime=regime,sub_regime=sub_regime,
				columns=sensitive_columns)

			# Fill out tree
			parse_tree.create_from_ast(constraint_str)
			# assign deltas for each base node
			# use equal weighting for each base node
			parse_tree.assign_deltas(weight_method='equal')

			# Assign bounds needed on the base nodes
			parse_tree.assign_bounds_needed()
			
			parse_trees.append(parse_tree)

		spec = SupervisedSpec(
			dataset=dataset,
			model=model,
			sub_regime="regression",
			frac_data_in_safety=0.6,
			primary_objective=primary_objective,
			parse_trees=parse_trees,
			initial_solution_fn=model.fit,
			use_builtin_primary_gradient_fn=True,
			optimization_technique='gradient_descent',
			optimizer='adam',
			optimization_hyperparams={
				'lambda_init'   : 0.5,
				'alpha_theta'   : 0.005,
				'alpha_lamb'    : 0.005,
				'beta_velocity' : 0.9,
				'beta_rmsprop'  : 0.95,
				'num_iters'     : 200,
				'gradient_library': "autograd",
				'hyper_search'  : None,
				'verbose'       : True,
			}
		)
		return spec
	
	yield spec_maker
	print("Teardown gpa_regression_spec")
	

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
		policy = Softmax(hyperparam_and_setting_dict={},
			env_description=env_description)
		env_kwargs={'gamma':0.9}
		save_dir = '.'

		spec = createRLSpec(
			dataset=dataset,
			policy=policy,
			constraint_strs=constraint_strs,
			deltas=deltas,
			env_kwargs=env_kwargs,
			save=True,
			save_dir='.',
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


