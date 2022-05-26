import os
import pickle
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

from seldonian.dataset import DataSet
from seldonian import seldonian_algorithm

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
		test_features = kwargs['test_features']
		test_labels = kwargs['test_labels']

		helper = partial(
			self.model_class_dict[self.model_name],
			test_features=test_features,
			test_labels=test_labels,
			dataset=kwargs['dataset'],
			constraint_funcs=kwargs['constraint_funcs'],
			eval_method=kwargs['eval_method'],
			results_dir=kwargs['results_dir'],
			max_iter=kwargs['max_iter'],
			verbose=kwargs['verbose'],
			n_jobs=kwargs['n_jobs'],
		)
		data_pcts = kwargs['data_pcts']
		n_trials = kwargs['n_trials']

		data_pcts_vector = np.array([x for x in data_pcts for y in range(n_trials)])
		trials_vector = np.array([x for y in range(len(data_pcts)) for x in range(n_trials)])
		
		# for data_pct in data_pcts:
		# 	for trial_i in range(n_trials):
		# 		print(data_pct,trial_i)
		# 		helper(data_pct,trial_i)

		with ProcessPoolExecutor(max_workers=kwargs['n_jobs']) as ex:
			results = tqdm(ex.map(helper,data_pcts_vector,trials_vector),
				total=len(data_pcts_vector))
			for exc in results:
				if exc:
					print(exc)
		self._aggregate_results(**kwargs)
	
	def logistic_regression(self,data_pct,trial_i,**kwargs):
		trial_dir = os.path.join(
				kwargs['results_dir'],
				'logistic_regression_results',
				'trial_data')

		os.makedirs(trial_dir,exist_ok=True)
		dataset = kwargs['dataset']
		label_column = dataset.label_column
		orig_df = dataset.df
		# number of points in this partition of the data
		n_points = int(round(data_pct*len(orig_df))) 

		test_features = kwargs['test_features']
		test_labels = kwargs['test_labels']
		
		if kwargs['eval_method'] == 'resample':
			resampled_filename = os.path.join(kwargs['results_dir'],
			'resampled_dataframes',f'trial_{trial_i}.pkl')
			with open(resampled_filename,'rb') as infile:
				df = pickle.load(infile).iloc[:n_points]
			# df = pd.read_csv(resampled_filename).iloc[:n_points]

		else:
			raise NotImplementedError(f"eval_method: {eval_method} not implemented")

		# Set up for initial solution
		labels = df[label_column]
		features = df.loc[:,
			df.columns != label_column]
		
		if not dataset.include_sensitive_columns:
			features = features.drop(
				columns=dataset.sensitive_column_names)

		if dataset.include_intercept_term:
			features.insert(0,'offset',1.0) # inserts a column of 1's
		
		lr_model = LogisticRegression(max_iter=kwargs['max_iter'])
		lr_model.fit(features, labels)
		theta_solution = np.hstack([lr_model.intercept_,lr_model.coef_[0]])
		# predict the class label, not the probability
		prediction_test = lr_model.predict(test_features) 
		
		# Calculate accuracy 
		acc = np.mean(1.0*prediction_test==test_labels)
		performance = acc
		print(f"Accuracy = {performance}")
		# Determine whether this solution
		# violates any of the constraints 
		# on the test dataset
		failed = False
		for parse_tree in parse_trees:
			parse_tree.evaluate_constraint(theta=theta_solution,
				dataset=dataset,
				model=lr_model,regime='supervised',
				branch='safety_test')

			ghat = parse_tree.root.value
			if ghat > 0:
				failed = True

		# Write out file for this data_pct,trial_i combo
		data = [data_pct,
			trial_i,
			performance,
			failed]
		colnames = ['data_pct','trial_i','performance','failed']
		self.write_trial_result(
			data,
			colnames,
			trial_dir,
			verbose=kwargs['verbose'])
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
				resampled_filename = os.path.join(kwargs['results_dir'],
				'resampled_dataframes',f'trial_{trial_i}.csv')
				n_points = int(round(data_pct*len(test_features))) 
				with open(resampled_filename,'rb') as infile:
					df = pickle.load(infile).iloc[:n_points]
				# df = pd.read_csv(resampled_filename).iloc[:n_points]
			else:
				raise NotImplementedError(f"eval_method: {eval_method} not implemented")
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
			performance = acc
			# Determine whether this solution
			# violates any of the constraints 
			# on the test dataset
			failed = False
			for gfunc in kwargs['constraint_funcs']:
				for parse_tree in parse_trees:
					parse_tree.evaluate_constraint(theta=candidate_solution,
						dataset=dataset,
						model=model,regime='supervised',
						branch='safety_test')
					ghat = parse_tree.root.value
					if ghat > 0:
						failed = True

			# Write out file for this data_pct,trial_i combo
			data = [data_pct,
				trial_i,
				performance,
				failed]
			colnames = ['data_pct','trial_i','performance','failed']
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
		n_jobs = kwargs['n_jobs']
		partial_kwargs = {key:kwargs[key] for key in kwargs \
			if key not in ['data_pcts','n_trials']}

		helper = partial(
			self.QSA,
			**partial_kwargs
		)

		data_pcts = kwargs['data_pcts']
		n_trials = kwargs['n_trials']
		data_pcts_vector = np.array([x for x in data_pcts for y in range(n_trials)])
		trials_vector = np.array([x for y in range(len(data_pcts)) for x in range(n_trials)])

		if n_jobs == 1:
			for ii in range(len(data_pcts_vector)):
				data_pct = data_pcts_vector[ii]
				trial_i = trials_vector[ii]
				print(data_pct,trial_i)
				helper(data_pct,trial_i)
		elif n_jobs > 1:
			with ProcessPoolExecutor(max_workers=n_jobs,
				mp_context=mp.get_context('fork')) as ex:
				results = tqdm(ex.map(helper,data_pcts_vector,trials_vector),
					total=len(data_pcts_vector))
				for exc in results:
					if exc:
						print(exc)
		else:
			raise ValueError(f"n_jobs value of {n_jobs} must be >=1 ")

		self._aggregate_results(**kwargs)
	
	def QSA(self,data_pct,trial_i,**kwargs):
		
		spec = kwargs['spec']
		verbose=kwargs['verbose']
		eval_method = kwargs['eval_method']
		perf_eval_fn = kwargs['perf_eval_fn']

		regime=spec.dataset.regime

		trial_dir = os.path.join(
				self.results_dir,
				'qsa_results',
				'trial_data')

		savename = os.path.join(trial_dir,
			f'data_pct_{data_pct:.4f}_trial_{trial_i}.csv')

		if os.path.exists(savename):
			if verbose:
				print(f"Trial {trial_i} already run for"
					  f"this data_pct: {data_pct}. Skipping this trial. ")
			return

		os.makedirs(trial_dir,exist_ok=True)
		
		parse_trees = spec.parse_trees
		dataset = spec.dataset

		if regime == 'supervised':
			# Load in ground truth
			test_features = kwargs['test_features']
			test_labels = kwargs['test_labels']

			sensitive_column_names = dataset.sensitive_column_names
			include_sensitive_columns = dataset.include_sensitive_columns
			include_intercept_term = dataset.include_intercept_term
			label_column = dataset.label_column

			if eval_method == 'resample':
				resampled_filename = os.path.join(self.results_dir,
					'resampled_dataframes',f'trial_{trial_i}.pkl')
				n_points = int(round(data_pct*len(test_features))) 

				with open(resampled_filename,'rb') as infile:
					resampled_df = pickle.load(infile).iloc[:n_points]

				if verbose:
					print(f"Using resampled dataset {resampled_filename} "
						  f"with {len(resampled_df)} datapoints")
			else:
				raise NotImplementedError(
					f"Eval method {eval_method} "
					f"not supported for regime={regime}")

			dataset_for_experiment = DataSet(
				df=resampled_df,
				meta_information=resampled_df.columns,
				regime=regime,
				label_column=label_column,
				sensitive_column_names=sensitive_column_names,
				include_sensitive_columns=include_sensitive_columns,
				include_intercept_term=include_intercept_term)

			# Make a new spec object where the 
			# only thing that is different is the dataset

			spec_for_experiment = copy.deepcopy(spec)
			spec_for_experiment.dataset = dataset_for_experiment
			model_instance = spec_for_experiment.model()

		elif regime == 'RL':
			RL_environment_obj = kwargs['RL_environment_obj']
			
			if eval_method == 'generate_episodes':
				# Sample from resampled dataset on disk of n_episodes
				save_dir = os.path.join(kwargs['results_dir'],'resampled_datasets')
				# savename = os.path.join(save_dir,f'resampled_df_trial{trial_i}.csv')
				savename = os.path.join(save_dir,f'resampled_df_trial{trial_i}.pkl')
				with open(savename,'rb') as infile:
					resampled_df = pickle.load(infile)

				# resampled_df = pd.read_csv(savename,names=dataset.df.columns[:-1])
				# Convert any array columns
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
					f"Eval method {eval_method} "
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
		

		# Run Seldonian algorithm with experimental spec
		passed_safety,candidate_solution = seldonian_algorithm(
			spec_for_experiment)
		print("Ran seldonian algorithm")
		NSF=False
		# Handle NSF 
		if type(candidate_solution) == str and candidate_solution == 'NSF':
			NSF = True
		
		# Calculate peformance
		
		failed=False # flag for whether we were actually safe 
		# on ground truth. Only relevant if we passed the safety test

		if NSF:
			performance = -99.0
		else:
			# If passed the safety test, calculate performance
			# using candidate solution 
			if passed_safety:
				if verbose:
					print("Passed safety test. Calculating performance")
				if regime == 'supervised':
					if verbose: 
						print("Calculating performance with candidate solution:")
						print(candidate_solution)
					
					performance = perf_eval_fn(
						model_instance,
						candidate_solution,test_features,
						test_labels)
					
					if verbose:
						print(f"Performance = {performance}")
				
				elif regime == 'RL':
					# Calculate J, the expected sum of discounted rewards
					# using this candidate solution on a bunch of newly 
					# generated episodes 
					RL_environment_obj.param_weights = candidate_solution
					# df_regen = RL_environment_obj.generate_data(
					# 	n_episodes=kwargs['n_episodes_for_eval'],
					# 	n_workers=kwargs['n_jobs'])
					df_regen = RL_environment_obj.generate_data(
						n_episodes=kwargs['n_episodes_for_eval'],
						n_workers=kwargs['n_jobs'],parallel=False)
					# print(df_regen)
					performance = RL_environment_obj.calc_J_from_df(df_regen,
						gamma=RL_environment_obj.gamma)
					print(f"Performance is J={performance}")
				
				# Calculate whether we failed on test set
				if verbose:
					print("Determining whether solution is actually safe on ground truth")
				
				if regime == 'supervised':
					for parse_tree in spec_for_experiment.parse_trees:
						parse_tree.evaluate_constraint(
							theta=candidate_solution,
							dataset=spec_for_experiment.dataset,
							model=model_instance,
							regime='supervised',
							branch='safety_test')
						
						ghat = parse_tree.root.value
						if ghat > 0:
							failed = True

						if verbose:
							constraint_str = parse_tree.constraint_str
							print(f"ghat for constraint: {constraint_str}"
								  f" on test set: {ghat}")

				elif regime == 'RL':
					ghat = gfunc(
						df_regen,
						RL_environment_obj=RL_environment_obj,
						)
					if ghat > 0:
						failed = True
				
				if verbose:
					if failed:
						print("Solution was not actually safe on test set!")
					else:
						print("Solution was safe on test set")
					print()
			else:
				if verbose:
					print("Failed safety test ")
					performance = -99.0
		
		# Reset param weights to their initial weights -- might not be neccesary?
		if regime == 'RL':
			RL_environment_obj.param_weights = RL_environment_obj.initial_weights

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
