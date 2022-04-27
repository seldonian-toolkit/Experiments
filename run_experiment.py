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
	parser.add_argument('--n_trials',   type=int,   default=10,   help='Number of trials to run')
	parser.add_argument('--data_pcts',   type=float,
		nargs='+',default=np.logspace(-2,0,num=15), 
		help='Proportions of the overall size of the dataset to use')

	parser.add_argument('--eval_method',   type=str, default='resample',
		help='How to use the dataset to evaluate performance, solution rate, failure rate')
	
	parser.add_argument('--n_episodes_for_eval',   type=int, default=10000,
		help='The number of episodes to use when evaluating the performance. RL-specific')
	
	parser.add_argument('--primary_objective',   type=str, default='default',
		help='String description of function to minimize. Will use model default if not provided')
	
	parser.add_argument('--RL_environment',   type=str,
		help='The name of the environment, e.g. "gridworld3x3" '
		'which needs to be a prefix of a .py file in the '
		'seldonian library containing the environment object')
	
	parser.add_argument('--frac_data_in_safety', type=float, default=0.6,
		help='Fraction of data in safety test')
	
	parser.add_argument('--seldonian_model_type', type=str, default='linear_classifier',
		help='Base model type to use for Seldonian models')
	
	parser.add_argument('--include_T',  action='store_true',
		help='Whether or not to include type as a predictive feature. Default False')
	
	parser.add_argument('--include_intercept_col', action='store_false', 
		help='Whether to add a column of ones as an offset as first feature column. Default True')
	
	parser.add_argument('--n_jobs',     type=int,   default=8,    help='Number of processes to use.')
	
	parser.add_argument('--optimizer',  type=str,   default='Nelder-Mead',
		help='Choice of optimizer to use in candidate selection.')
	
	parser.add_argument('--max_iter',  type=int,   default=200,
		help='Max number of iterations of optimizer')
	
	parser.add_argument('--baseline',  type=str,   default='logistic_regression',
		help='The standard machine learning algorithm to compare against')
	
	parser.add_argument('--results_dir',  type=dir_path,   default='results',
		help='The directory where you want all of your results to be saved')
	
	parser.add_argument('--verbose',  action='store_true',
		help='Whether you want logs displayed. Default False')

	args = parser.parse_args()
	print(args.__dict__)
	# print(vars(args))
	extra_seldonian_kwargs = {}
	extra_baseline_kwargs = {}

	# Load dataset from interface output folder
	ds_pth = os.path.join(args.interface_output_pth,'dataset.p')
	with open(ds_pth,'rb') as infile:
		dataset = pickle.load(infile)
	print(dataset.meta_information)
	regime = dataset.regime

	if regime == 'supervised':	
		include_T=args.include_T,
		include_intercept_col=args.include_intercept_col,
		extra_baseline_kwargs['include_T'] = include_T
		extra_baseline_kwargs['include_intercept_col'] = include_intercept_col
		extra_seldonian_kwargs['include_T'] = include_T
		extra_seldonian_kwargs['include_intercept_col'] = include_intercept_col
		label_column = dataset.label_column
		sensitive_column_names = dataset.sensitive_column_names
		if args.eval_method == 'resample':
			# Use entire original dataset as ground truth for test set
			test_features = dataset.df.loc[:,
				dataset.df.columns != label_column]

			test_features = test_features.drop(
				columns=sensitive_column_names)
			test_labels = dataset.df[label_column]
			extra_baseline_kwargs['test_features'] = test_features
			extra_baseline_kwargs['test_labels'] = test_labels
			extra_seldonian_kwargs['test_features'] = test_features
			extra_seldonian_kwargs['test_labels'] = test_labels
	elif regime == 'RL':
		# Get enviornment and pass it to kwargs
		RL_environment_name = args.RL_environment
		RL_environment_module = importlib.import_module(
		f'seldonian.RL.environments.{RL_environment_name}')
		RL_environment_obj = RL_environment_module.Environment()	
		extra_seldonian_kwargs['RL_environment_obj'] = RL_environment_obj
		extra_seldonian_kwargs['n_episodes_for_eval'] = args.n_episodes_for_eval
		
		# generate full-size datasets for each trial so that 
		# I can reference them for each data_pct
		save_dir = os.path.join(args.results_dir,'resampled_datasets')
		os.makedirs(save_dir,exist_ok=True)
		print("generating resampled datasets")
		for trial_i in range(args.n_trials):
			savename = os.path.join(save_dir,f'resampled_df_trial{trial_i}.csv')
			if not os.path.exists(savename):
				RL_environment_obj.generate_data(n_episodes=args.n_episodes_for_eval,
					n_workers=args.n_jobs,savename=savename)
	### Get constraint functions from file
	constraint_module = importlib.import_module(
		os.path.basename(args.constraint_filename).split('.')[0])
	constraint_funcs = constraint_module.constraints
	if hasattr(constraint_module,'precalc'):
		constraints_precalc_func = constraint_module.precalc
	else:
		constraints_precalc_func = None
	# ## Setup for baseline experiments
	# run_baseline_kwargs = dict(
	# 	dataset=dataset,
	# 	data_pcts=args.data_pcts,
	# 	n_trials=args.n_trials,
	# 	constraint_funcs=constraint_funcs,
	# 	constraints_precalc_func=constraints_precalc_func,
	# 	n_jobs=args.n_jobs,
	# 	max_iter=args.max_iter,
	# 	eval_method=args.eval_method,
	# 	results_dir=args.results_dir,
	# 	verbose=args.verbose,
	# 	)

	# # Add any extra kwargs
	# for key in extra_baseline_kwargs:
	# 	run_baseline_kwargs[key] = extra_baseline_kwargs[key]

	# ## Run baseline model 
	# bl_exp = BaselineExperiment(model_name=args.baseline,
	# 	results_dir=args.results_dir)

	# bl_exp.run_experiment(**run_baseline_kwargs)

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
		constraints_precalc_func=constraints_precalc_func,
		frac_data_in_safety=args.frac_data_in_safety,
		seldonian_model_type=args.seldonian_model_type,
		n_jobs=args.n_jobs,
		max_iter=args.max_iter,
		eval_method=args.eval_method,
		primary_objective=args.primary_objective,
		results_dir=args.results_dir,
		optimizer=args.optimizer,
		regime=regime,
		verbose=args.verbose,
		)

	# Add any extra kwargs
	for key in extra_seldonian_kwargs:
		run_seldonian_kwargs[key] = extra_seldonian_kwargs[key]
	print(run_seldonian_kwargs)
	# ## Run Seldonian model 
	sd_exp = SeldonianExperiment(results_dir=args.results_dir)

	sd_exp.run_experiment(**run_seldonian_kwargs)

	
