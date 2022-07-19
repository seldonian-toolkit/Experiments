import os
import shutil

import pytest

from seldonian.models.models import LinearRegressionModel
from seldonian.utils.io_utils import load_json
from seldonian.dataset import DataSetLoader
from seldonian.parse_tree.parse_tree import ParseTree
from seldonian.spec import SupervisedSpec

@pytest.fixture
def gpa_regression_spec():
	print("Setup gpa_regression_spec")
	
	def spec_maker(constraint_strs,deltas):

		data_pth = 'static/datasets/supervised/GPA/gpa_regression_dataset.csv'
		metadata_pth = 'static/datasets/supervised/GPA/metadata_regression.json'

		metadata_dict = load_json(metadata_pth)
		regime = metadata_dict['regime']
		sub_regime = metadata_dict['sub_regime']
		columns = metadata_dict['columns']
		sensitive_columns = metadata_dict['sensitive_columns']
					
		include_sensitive_columns = False
		include_intercept_term = True
		regime='supervised'

		model_class = LinearRegressionModel

		# Mean squared error
		primary_objective = model_class().default_objective

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
				regime='supervised',sub_regime='regression',
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
			model_class=model_class,
			frac_data_in_safety=0.6,
			primary_objective=primary_objective,
			parse_trees=parse_trees,
			initial_solution_fn=model_class().fit,
			use_builtin_primary_gradient_fn=True,
			bound_method='ttest',
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
def experiment(request):
	results_dir = request.param
	""" Fixture to create and then remove results_dir and any files it may contain"""
	print("Setup experiment_cleanup")
	os.makedirs(results_dir,exist_ok=True)
	yield
	print("Teardown experiment_cleanup")
	shutil.rmtree(results_dir)


