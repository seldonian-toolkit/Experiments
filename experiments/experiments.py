import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

from seldonian.dataset import DataSet
from seldonian.model import *
from seldonian.candidate_selection import CandidateSelection
from seldonian.safety_test import SafetyTest


class Experiment():
	def _aggregate_results(self,**kwargs):

		savedir_results = os.path.join(
			self.results_dir,
			f'{self.model_name}_results')
		os.makedirs(savedir_results,exist_ok=True)
		savename_results = os.path.join(savedir_results,
			f'{self.model_name}_results.csv')
		
		trial_dir = os.path.join(
			self.results_dir,
			f'{self.model_name}_results',
			'trial_data')
		df_list = []
		for data_pct in kwargs['data_pcts']:
			for trial_i in range(kwargs['n_trials']):
				filename = os.path.join(trial_dir,
					f'data_pct_{data_pct:.4f}_trial_{trial_i}.csv')
				df = pd.read_csv(filename)
				df_list.append(df)

		result_df = pd.concat(df_list)
		result_df.to_csv(savename_results,index=False)
		print(f"Saved {savename_results}")
		return

	def write_trial_result(self,data,
		colnames,trial_dir, 
		verbose=False):
		result_df = pd.DataFrame([data])
		result_df.columns = colnames
		data_pct,trial_i = data[0:2]

		savename = os.path.join(trial_dir,
			f'data_pct_{data_pct:.4f}_trial_{trial_i}.csv')

		result_df.to_csv(savename,index=False)
		if verbose:
			print(f"Saved {savename}")
		return

class BaselineExperiment(Experiment):
	def __init__(self,model_name,results_dir):
		self.model_name = model_name
		self.model_class_dict = {
			'logistic_regression':self.logistic_regression,
			'linear_svm':self.linear_svm}
		self.results_dir = results_dir

	def run_experiment(self,**kwargs):
		# Do any data precalculation 
		if 'constraints_precalc_func' in kwargs:
			precalc_dict = kwargs['constraints_precalc_func'](kwargs['dataset'])
		else:
			precalc_dict = {}

		test_features = kwargs['test_features']
		if self.model_name in ['logistic_regression','linear_svm']:
			test_labels = kwargs['test_labels'].copy()
			test_labels.loc[test_labels==-1]=0
		else:
			test_labels = kwargs['test_labels']

		helper = partial(
			self.model_class_dict[self.model_name],
			test_features=test_features,
			test_labels=test_labels,
			dataset=kwargs['dataset'],
			constraint_funcs=kwargs['constraint_funcs'],
			eval_method=kwargs['eval_method'],
			precalc_dict=precalc_dict,
			results_dir=kwargs['results_dir'],
			max_iter=kwargs['max_iter'],
			verbose=kwargs['verbose'],
			n_jobs=kwargs['n_jobs'],
		)
		data_pcts = kwargs['data_pcts']
		n_trials = kwargs['n_trials']
		data_pcts_vector = np.array([x for x in data_pcts for y in range(n_trials)])
		trials_vector = np.array([x for y in range(len(data_pcts)) for x in range(n_trials)])
		
		for data_pct in data_pcts:
			for trial_i in range(n_trials):
				print(data_pct,trial_i)
				helper(data_pct,trial_i)

		# with ProcessPoolExecutor(max_workers=kwargs['n_jobs']) as ex:
		# 	results = tqdm(ex.map(helper,data_pcts_vector,trials_vector),
		# 		total=len(data_pcts_vector))
		# 	for exc in results:
		# 		if exc:
		# 			print(exc)
		self._aggregate_results(**kwargs)
	
	def logistic_regression(self,data_pct,trial_i,**kwargs):
		try: 
			trial_dir = os.path.join(
					kwargs['results_dir'],
					'logistic_regression_results',
					'trial_data')

			os.makedirs(trial_dir,exist_ok=True)
			dataset = kwargs['dataset']
			orig_df = dataset.df
			# number of points in this partition of the data
			n_points = int(round(data_pct*len(orig_df))) 

			test_features = kwargs['test_features']
			test_labels = kwargs['test_labels']
			
			if kwargs['eval_method'] == 'resample':
				df = orig_df.sample(n=len(orig_df),replace=True)[0:n_points]

			# set labels to 0 and 1, not -1 and 1, like some classification datasets will have
			df.loc[df[dataset.label_column]==-1.0,dataset.label_column]=0.0

			features = df.loc[:,
				df.columns != dataset.label_column]

			features = features.drop(
				columns=dataset.sensitive_column_names)

			labels = df[dataset.label_column]
			
			lr_model = LogisticRegression(max_iter=kwargs['max_iter'])
			lr_model.fit(features, labels)

			prediction = lr_model.predict(test_features)
			
			# Calculate accuracy 
			acc = np.mean(1.0*prediction==test_labels)

			# Determine whether this solution
			# violates any of the constraints 
			# on the test dataset
			failed = False
			for gfunc in kwargs['constraint_funcs']:
				ghat = gfunc(
					test_features,
					test_labels,
					prediction,
					precalc_dict=kwargs['precalc_dict'])
				if ghat > 0:
					failed = True

			# Write out file for this data_pct,trial_i combo
			data = [data_pct,
				trial_i,
				acc,
				failed]
			colnames = ['data_pct','trial_i','acc','failed']
			self.write_trial_result(
				data,
				colnames,
				trial_dir,
				verbose=kwargs['verbose'])
		except Exception as e:
			return e
		return None

	def linear_svm(self,data_pct,trial_i,**kwargs):
		try: 
			trial_dir = os.path.join(
					kwargs['results_dir'],
					'linear_svm_results',
					'trial_data')

			os.makedirs(trial_dir,exist_ok=True)
			dataset = kwargs['dataset']
			orig_df = dataset.df
			# number of points in this partition of the data
			n_points = int(round(data_pct*len(orig_df))) 

			test_features = kwargs['test_features']
			test_labels = kwargs['test_labels']
			
			if kwargs['eval_method'] == 'resample':
				df = orig_df.sample(n=len(orig_df),replace=True)[0:n_points]

			# set labels to 0 and 1, not -1 and 1, like some classification datasets will have
			df.loc[df[dataset.label_column]==-1.0,dataset.label_column]=0.0

			features = df.loc[:,
				df.columns != dataset.label_column]

			features = features.drop(
				columns=dataset.sensitive_column_names)

			labels = df[dataset.label_column]
			
			clf = make_pipeline(StandardScaler(),
			  SGDClassifier(loss='hinge',max_iter=kwargs['max_iter']))
			
			clf.fit(features, labels)

			prediction = clf.predict(test_features)
			
			# Calculate accuracy 
			acc = np.mean(1.0*prediction==test_labels)

			# Determine whether this solution
			# violates any of the constraints 
			# on the test dataset
			failed = False
			for gfunc in kwargs['constraint_funcs']:
				ghat = gfunc(
					test_features,
					test_labels,
					prediction,
					precalc_dict=kwargs['precalc_dict'])
				if ghat > 0:
					failed = True

			# Write out file for this data_pct,trial_i combo
			data = [data_pct,
				trial_i,
				acc,
				failed]
			colnames = ['data_pct','trial_i','acc','failed']
			self.write_trial_result(
				data,
				colnames,
				trial_dir,
				verbose=kwargs['verbose'])
		except Exception as e:
			return e
		return None

class SeldonianExperiment(Experiment):
	def __init__(self,results_dir):
		self.results_dir = results_dir
		self.model_name = 'qsa'

	def run_experiment(self,**kwargs):
		# Do any data precalculation 
		if 'precalc_func' in kwargs:
			precalc_dict = kwargs['constraints_precalc_func'](kwargs['dataset'])
		else:
			precalc_dict = {}
		
		partial_kwargs = {key:kwargs[key] for key in kwargs \
			if key not in ['data_pcts','n_trials']}
		partial_kwargs['precalc_dict'] = precalc_dict
	
		helper = partial(
			self.QSA,
			**partial_kwargs
		)

		data_pcts = kwargs['data_pcts']
		n_trials = kwargs['n_trials']
		data_pcts_vector = np.array([x for x in data_pcts for y in range(n_trials)])
		trials_vector = np.array([x for y in range(len(data_pcts)) for x in range(n_trials)])
		# for ii in range(len(data_pcts_vector)):
		# 	data_pct = data_pcts_vector[ii]
		# 	trial_i = trials_vector[ii]
		# 	print(data_pct,trial_i)
		# 	helper(data_pct,trial_i)
		with ProcessPoolExecutor(max_workers=kwargs['n_jobs']) as ex:
			results = tqdm(ex.map(helper,data_pcts_vector,trials_vector),total=len(data_pcts_vector))
			for exc in results:
				if exc:
					print(exc)
		self._aggregate_results(**kwargs)
	
	def QSA(self,data_pct,trial_i,**kwargs):
		# try: 
		verbose=kwargs['verbose']
		regime=kwargs['regime']
		frac_data_in_safety = kwargs['frac_data_in_safety']

		trial_dir = os.path.join(
				kwargs['results_dir'],
				'qsa_results',
				'trial_data')

		os.makedirs(trial_dir,exist_ok=True)
		
		parse_trees = kwargs['parse_trees']
		dataset = kwargs['dataset']
		orig_df = dataset.df

		if regime == 'supervised':
			# number of points in this partition of the data
			n_points = int(round(data_pct*len(orig_df))) 
			test_features = kwargs['test_features']
			test_labels = kwargs['test_labels']
			sensitive_column_names = dataset.sensitive_column_names
			label_column = dataset.label_column

			if kwargs['eval_method'] == 'resample':
				# use random state by trial so that each 
				# trial always has the same resampled data
				# then just get the first n_points 
				df = orig_df.sample(n=len(orig_df),
					replace=True,random_state=trial_i)[0:n_points]
			else:
				raise NotImplementedError(
					f"Eval method {kwargs['eval_method']} "
					"not supported for regime={regime}")

			candidate_df, safety_df = train_test_split(
				df, test_size=frac_data_in_safety, shuffle=False)

			sensitive_candidate_df = candidate_df[sensitive_column_names]
			sensitive_safety_df = safety_df[sensitive_column_names]
			candidate_dataset = DataSet(
				candidate_df,meta_information=df.columns,
				sensitive_column_names=sensitive_column_names,
				regime=regime,label_column=label_column)

			safety_dataset = DataSet(
				safety_df,meta_information=df.columns,
				sensitive_column_names=sensitive_column_names,
				regime=regime,label_column=label_column)

			n_safety = len(safety_df)

			# Set up initial solution
			labels = candidate_dataset.df[label_column]
			features = candidate_dataset.df.loc[:,
				candidate_dataset.df.columns != label_column]
			if not include_sensitive_columns:
				features = features.drop(
					columns=candidate_dataset.sensitive_column_names)
			if include_intercept_term:
				features.insert(0,'offset',1.0) # inserts a column of 1's
			initial_solution = model_instance.fit(features,labels)

		elif regime == 'RL':
			RL_environment_obj = kwargs['RL_environment_obj']
			
			if kwargs['eval_method'] == 'generate_episodes':
				# Sample from resampled dataset on disk of n_episodes
				save_dir = os.path.join(kwargs['results_dir'],'resampled_datasets')
				savename = os.path.join(save_dir,f'resampled_df_trial{trial_i}.csv')
				resampled_df = pd.read_csv(savename,names=dataset.df.columns[:-1])
				n_episodes_max = resampled_df['episode_index'].nunique()

				n_episodes = int(round(n_episodes_max*data_pct))
				print(f"Orig dataset should have {n_episodes_max} episodes")
				print(f"This dataset with data_pct={data_pct} should have {n_episodes} episodes")
				
				# Take first n_episodes episodes 

				resampled_df = resampled_df.loc[resampled_df['episode_index']<n_episodes]
				resampled_episodes = resampled_df.episode_index.unique()
				# For candidate take first 1.0-frac_data_in_safety fraction
				# and for safety take remaining
				n_candidate = int(round(n_episodes*(1.0-frac_data_in_safety)))
				candidate_episodes = resampled_episodes[0:n_candidate]
				safety_episodes = resampled_episodes[n_candidate:]
				
				safety_df = resampled_df.copy().loc[
					resampled_df['episode_index'].isin(safety_episodes)]
				candidate_df = resampled_df.copy().loc[
					resampled_df['episode_index'].isin(candidate_episodes)]

				print("Safety dataset has n_episodes:")
				print(safety_df['episode_index'].nunique())
				print("Candidate dataset has n_episodes:")
				print(candidate_df['episode_index'].nunique())
				# print(candidate_df)
			else:
				raise NotImplementedError(
					f"Eval method {kwargs['eval_method']} "
					"not supported for regime={regime}")
			
			candidate_dataset = DataSet(
				candidate_df,meta_information=resampled_df.columns,
				regime=regime)

			safety_dataset = DataSet(
				safety_df,meta_information=resampled_df.columns,
				regime=regime)

			n_safety = safety_df['episode_index'].nunique()
			n_candidate = candidate_df['episode_index'].nunique()
			print(f"Safety dataset has {n_safety} episodes")
			print(f"Candidate dataset has {n_candidate} episodes")
			# input("wait!")
			# Set up initial solution
			initial_solution = RL_environment_obj.initial_weights
			print(f"Initial solution: {initial_solution}")
		
		# Determine model to use
		if kwargs['seldonian_model_type'] == 'linear_classifier':
			model = LinearClassifierModel()

		if kwargs['seldonian_model_type'] == 'tabular_softmax':
			model = TabularSoftmaxModel(RL_environment_obj)

		# Candidate selection
		if kwargs['primary_objective'] == 'default':
			primary_objective = model.default_objective
		else:
			primary_objective_str = kwargs['primary_objective']
			if primary_objective_str == 'perceptron_loss':
				primary_objective = model.perceptron_loss
		print(primary_objective)
		cs = CandidateSelection(
			model=model,
			candidate_dataset=candidate_dataset,
			n_safety=n_safety,
			parse_trees=parse_trees,
			primary_objective=primary_objective,
			optimization_technique='barrier_function',
			optimizer=kwargs['optimizer'],
			initial_solution=initial_solution,
			regime=regime)

		minimizer_options = {'maxiter':kwargs['max_iter']}
		print("Running candidate selection")
		print()
		candidate_solution = cs.run(minimizer_options=minimizer_options)
		# candidate_solution = np.array([ 0.24572697,  0.03441104, -0.10560843, -0.03138242, -0.1230764,  -0.24395811,
		#   0.04524236, -0.02241922, -0.03385316,  0.19522918, -0.26457132, -0.07821181,
		#  -0.0280935,   0.55131979,  0.354055,    0.19880951, -0.01729549, -0.04897922,
		#   0.17336506,  0.40439311,  0.12206978,  0.04952516, -0.12059848,  0.1056631,
		#   0.07874577, -0.11638033, -0.42527073, -0.20060874, -0.15377955, -0.28476562,
		#   0.09273259,  0.19022399])
		# candidate_solution = np.array([-10.0,-10.0,-10.0,10.0,-10.0,-10.0,-10.0,10.0,
  #           -10.0,10.0,-10.0,-10.0,-10.0,-10.0,-10.0,10.0,-10.0,-10.0,-10.0,10.0,-10.0,10.0,-10.0,-10.0,
  #           10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,10.0])
		# candidate_solution = np.array([0.48904387, -0.95299081, -0.6351963, 0.96031704,-0.25886638, 0.00390796, -0.97168071,
		# 	0.87999303,  0.3201818,   0.7680256,  -0.19126927,  0.08804582, 0.46200017, -0.26357572,  0.12584299, -0.57352906,  0.11140123, -1.09147967,
		# 	-0.43219567,  0.70882718, -1.22288815,  0.54908985, -0.73525076,  1.03962311,
		# 	-0.27383999,  0.46052568,  0.48903373,  0.39273773, -0.18736683,  0.12802126,
		# 	-0.12181113,  1.03362185])
		# Safety test
		st = SafetyTest(safety_dataset,
			model,
			parse_trees,
			regime=regime)

		passed_safety = st.run(candidate_solution,
			bound_method='ttest')

		print(passed_safety)
		print(candidate_solution)
		# If passed the safety test, calculate performance
		# using candidate solution 
		if passed_safety:
			print("Passed safety test. Calculating performance")
			if regime == 'supervised':
				prediction = model.predict(
					candidate_solution,
					test_features)

				acc = np.mean(1.0*prediction==test_labels)
				performance = acc
			elif regime == 'RL':
				# Calculate J, the expected sum of discounted rewards
				# using this candidate solution on a bunch of newly 
				# generated episodes 
				RL_environment_obj.param_weights = candidate_solution
				df_regen = RL_environment_obj.generate_data(
					n_episodes=kwargs['n_episodes_for_eval'])
				# print(df_regen)
				performance = RL_environment_obj.calc_J_from_df(df_regen,
					gamma=RL_environment_obj.gamma)
				print(f"Performance is J={performance}")
		else:
			print("Failed safety test")
			performance = -99.0

		# Calculate whether we failed, i.e. 
		# we said we passed the safety test but
		# the constraint fails on the original dataset (ground truth)
		failed=False
		if passed_safety:
			print("Determining whether solution is actually safe")
			for gfunc in kwargs['constraint_funcs']:
				if regime == 'supervised':
					ghat = gfunc(
						test_features,
						test_labels,
						prediction,
						precalc_dict=kwargs['precalc_dict'])
				elif regime == 'RL':
					ghat = gfunc(
						n_episodes_for_eval=kwargs['n_episodes_for_eval'],
						param_weights=candidate_solution,
						RL_environment_obj=RL_environment_obj,
						)
				if ghat > 0:
					failed = True

		# Write out file for this data_pct,trial_i combo
		data = [data_pct,
			trial_i,
			performance,
			passed_safety,
			failed]
		colnames = ['data_pct','trial_i','performance','passed_safety','failed']
		self.write_trial_result(
			data,
			colnames,
			trial_dir,
			verbose=kwargs['verbose'])
		return
		# except Exception as e:
		# 	return e
		# return None
