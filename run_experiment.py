import os
import glob
import pickle
import importlib
import numpy as np
import argparse

from seldonian.dataset import DataSetLoader
from seldonian.io_utils import dir_path

from experiments.experiments import (
	BaselineExperiment,SeldonianExperiment)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('interface_output_pth',   type=dir_path, help='Path to output folder from running interface')
	parser.add_argument('constraint_filename', type=str,   help='filename containing constraint function(s)')
	parser.add_argument('--eval_method',   type=str, default='resample',
		help='How to use the dataset to evaluate accuracy, solution rate, failure rate')
	parser.add_argument('--n_trials',   type=int,   default=10,   help='Number of trials to run')
	parser.add_argument('--data_pcts',   type=float,
		nargs='+',default=np.logspace(-2,0,num=15), 
		help='Proportions of the overall size of the dataset to use')
	parser.add_argument('--frac_data_in_safety', type=float, default=0.6,
		help='Fraction of data in safety test')
	parser.add_argument('--seldonian_model_type', type=str, default='linear_classifier',
		help='Base model type to use for Seldonian models')
	parser.add_argument('--include_T',  action='store_true',
		help='Whether or not to include type as a predictive feature')
	parser.add_argument('--n_jobs',     type=int,   default=8,    help='Number of processes to use.')
	parser.add_argument('--optimizer',  type=str,   default='Nelder-Mead',
		help='Choice of optimizer to use in candidate selection.')
	parser.add_argument('--baseline',  type=str,   default='logistic_regression',
		help='The standard machine learning algorithm to compare against')
	parser.add_argument('--include_intercept_col',  type=bool, default=True, 
		help='Whether to add a column of ones as an offset as first feature column')
	parser.add_argument('--results_dir',  type=dir_path,   default='results',
		help='The directory where you want all of your results to be saved')
	parser.add_argument('--verbose',  type=int,   default=0,
		help='Level of verbosity you want. 0 is no logs, 1 is full logging.')

	args = parser.parse_args()

	# Load dataset from interface output folder
	ds_pth = os.path.join(args.interface_output_pth,'dataset.p')
	with open(ds_pth,'rb') as infile:
		dataset = pickle.load(infile)
		label_column = dataset.label_column
		sensitive_column_names = dataset.sensitive_column_names

	### Get constraint functions from file
	constraint_module = importlib.import_module(
		os.path.basename(args.constraint_filename).split('.')[0])
	constraint_funcs = constraint_module.constraints

	constraints_precalc_func = constraint_module.precalc

	if args.eval_method == 'resample':
		# Use entire original dataset as ground truth for test set
		test_features = dataset.df.loc[:,
			dataset.df.columns != label_column]

		test_features = test_features.drop(
			columns=sensitive_column_names)
		test_labels = dataset.df[label_column]

	## Setup for baseline experiments
	run_baseline_kwargs = dict(
		dataset=dataset,
		data_pcts=args.data_pcts,
		n_trials=args.n_trials,
		constraint_funcs=constraint_funcs,
		constraints_precalc_func=constraints_precalc_func,
		include_T=args.include_T,
		n_jobs=args.n_jobs,
		max_iter=1000,
		test_features=test_features,
		test_labels=test_labels,
		eval_method=args.eval_method,
		results_dir=args.results_dir,
		verbose=args.verbose,
		)

	## Run baseline model 
	bl_exp = BaselineExperiment(model_name=args.baseline,
		results_dir=args.results_dir)

	bl_exp.run_experiment(**run_baseline_kwargs)

	## Setup for Seldonian experiments
	# Need parse trees from interface output folder
	parse_tree_paths = glob.glob(os.path.join(args.interface_output_pth,'parse_tree*'))
	parse_trees = []
	for pth in parse_tree_paths:
		with open(pth,'rb') as infile:
			pt = pickle.load(infile)
			parse_trees.append(pt)
	
	run_seldonian_kwargs = dict(
		dataset=dataset,
		data_pcts=args.data_pcts,
		n_trials=args.n_trials,
		parse_trees=parse_trees,
		constraint_funcs=constraint_funcs,
		precalc_func=constraints_precalc_func,
		constraints_precalc_func=constraints_precalc_func,
		frac_data_in_safety=args.frac_data_in_safety,
		seldonian_model_type=args.seldonian_model_type,
		include_T=args.include_T,
		n_jobs=args.n_jobs,
		max_iter=200,
		test_features=test_features,
		test_labels=test_labels,
		eval_method=args.eval_method,
		results_dir=args.results_dir,
		optimizer=args.optimizer,
		include_intercept_col=args.include_intercept_col,
		verbose=args.verbose,
		)

	# ## Run Seldonian model 
	sd_exp = SeldonianExperiment(results_dir=args.results_dir)

	sd_exp.run_experiment(**run_seldonian_kwargs)

	
