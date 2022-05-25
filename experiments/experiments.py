import os
import pickle
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
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
		# for ii in range(len(data_pcts_vector)):
		# 	data_pct = data_pcts_vector[ii]
		# 	trial_i = trials_vector[ii]
		# 	print(data_pct,trial_i)
		# 	helper(data_pct,trial_i)

		with ProcessPoolExecutor(max_workers=kwargs['n_jobs'],mp_context=mp.get_context('fork')) as ex:
			results = tqdm(ex.map(helper,data_pcts_vector,trials_vector),
				total=len(data_pcts_vector))
			for exc in results:
				if exc:
					print(exc)
		self._aggregate_results(**kwargs)
	
	def QSA(self,data_pct,trial_i,**kwargs):
		# try: 
		verbose=kwargs['verbose']
		regime=kwargs['regime']
		frac_data_in_safety = kwargs['frac_data_in_safety']
		initializer=kwargs['initializer']
		optimization_technique=kwargs['optimization_technique']
		optimizer=kwargs['optimizer']

		trial_dir = os.path.join(
				kwargs['results_dir'],
				'qsa_results',
				'trial_data')
		savename = os.path.join(trial_dir,
			f'data_pct_{data_pct:.4f}_trial_{trial_i}.csv')
		if os.path.exists(savename):
			print(f"Trial {trial_i} already run for"
				   f"this data_pct: {data_pct}. Skipping this trial. ")
			return

		os.makedirs(trial_dir,exist_ok=True)
		
		parse_trees = kwargs['parse_trees']
		dataset = kwargs['dataset']

		if regime == 'supervised':
			# Load in ground truth
			test_features = kwargs['test_features']
			test_labels = kwargs['test_labels']

			sensitive_column_names = dataset.sensitive_column_names
			include_sensitive_columns = dataset.include_sensitive_columns
			include_intercept_term = dataset.include_intercept_term
			label_column = dataset.label_column

			if kwargs['eval_method'] == 'resample':
				# resampled_filename = os.path.join(kwargs['results_dir'],
				# 	'resampled_dataframes',f'trial_{trial_i}.csv')
				resampled_filename = os.path.join(kwargs['results_dir'],
					'resampled_dataframes',f'trial_{trial_i}.pkl')
				n_points = int(round(data_pct*len(test_features))) 
				# df = pd.read_csv(resampled_filename).iloc[:n_points]
				with open(resampled_filename,'rb') as infile:
					resampled_df = pickle.load(infile).iloc[:n_points]
				print(f"Using resampled dataset {resampled_filename} with {len(resampled_df)} datapoints")
			else:
				raise NotImplementedError(
					f"Eval method {kwargs['eval_method']} "
					"not supported for regime={regime}")

			candidate_df, safety_df = train_test_split(
				resampled_df, test_size=frac_data_in_safety, shuffle=False)

			sensitive_candidate_df = candidate_df[sensitive_column_names]
			sensitive_safety_df = safety_df[sensitive_column_names]
			
			candidate_dataset = DataSet(
				candidate_df,meta_information=resampled_df.columns,
				sensitive_column_names=sensitive_column_names,
				include_sensitive_columns=include_sensitive_columns,
				include_intercept_term=include_intercept_term,
				regime=regime,label_column=label_column)

			safety_dataset = DataSet(
				safety_df,meta_information=resampled_df.columns,
				sensitive_column_names=sensitive_column_names,
				include_sensitive_columns=include_sensitive_columns,
				include_intercept_term=include_intercept_term,
				regime=regime,label_column=label_column)

			n_safety = len(safety_df)
			print(f"Candidate dataset of length: {len(candidate_df)}")
			print(f"Safety dataset of length: {len(safety_df)}")
			# Set up for initial solution
			candidate_labels = candidate_dataset.df[label_column]
			candidate_features = candidate_dataset.df.loc[:,
				candidate_dataset.df.columns != label_column]
			
			if not include_sensitive_columns:
				candidate_features = candidate_features.drop(
					columns=candidate_dataset.sensitive_column_names)

			if include_intercept_term:
				candidate_features.insert(0,'offset',1.0) # inserts a column of 1's

		elif regime == 'RL':
			RL_environment_obj = kwargs['RL_environment_obj']
			
			if kwargs['eval_method'] == 'generate_episodes':
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
		

		# Determine model to use
		if kwargs['seldonian_model_type'] == 'linear_classifier':
			model = LinearClassifierModel()

		elif kwargs['seldonian_model_type'] == 'logistic_regression':
			model = LogisticRegressionModel()

		elif kwargs['seldonian_model_type'] == 'tabular_softmax':
			model = TabularSoftmaxModel(RL_environment_obj)

		elif kwargs['seldonian_model_type'] == 'linear_softmax':
			model = LinearSoftmaxModel(RL_environment_obj)
		
		## Initial solution
		if regime == 'supervised':
			if initializer == 'random':
				initial_solution = np.random.normal(0,1,candidate_features.shape[1])
			elif initializer == 'zeros':
				initial_solution = np.zeros(candidate_features.shape[1])
			elif initializer == 'fit':
				try:
					initial_solution = model.fit(candidate_features,candidate_labels)
				except ValueError:
					# If there is not enough data for a fit then use random init
					initial_solution = np.random.normal(0,1,candidate_features.shape[1])
			else:
				raise NotImplementedError(f"Initialization method: {initializer} not supported")
		elif regime == 'RL':
			initial_solution = RL_environment_obj.initial_weights

		print(f"initial solution is: {initial_solution}")

		# Candidate selection
		if kwargs['primary_objective'] == 'default':
			primary_objective = model.default_objective
		else:
			primary_objective_str = kwargs['primary_objective']
			if primary_objective_str == 'perceptron_loss':
				primary_objective = model.sample_perceptron_loss
			elif primary_objective_str == 'logistic_loss':
				primary_objective = model.sample_logistic_loss

		cs_kwargs = dict(
			model=model,
			candidate_dataset=candidate_dataset,
			n_safety=n_safety,
			parse_trees=parse_trees,
			primary_objective=primary_objective,
			optimization_technique=optimization_technique,
			optimizer=optimizer,
			initial_solution=initial_solution,
			regime=regime,
			write_logfile=True)
		
		if regime == 'RL':
			cs_kwargs['gamma'] = RL_environment_obj.gamma
			cs_kwargs['min_return'] = RL_environment_obj.min_return
			cs_kwargs['max_return'] = RL_environment_obj.max_return

		if 'reg_coef' in kwargs:
			cs_kwargs['reg_coef'] = kwargs['reg_coef']

		cs = CandidateSelection(**cs_kwargs)

		minimizer_options = {'maxiter':kwargs['max_iter']}
		cs_run_kwargs = dict(
			minimizer_options=minimizer_options,
			verbose=kwargs['verbose']
		)
		
		print("Running candidate selection")
		print()

		if optimization_technique == 'gradient_descent':
			cs_run_kwargs['use_primary_gradient'] = kwargs['use_primary_gradient']

		candidate_solution = cs.run(**cs_run_kwargs)

		# print()
		# print("################################################")
		# print("USING HARDCODED CANDIDATE SOLUTION FOR TESTING!!!")
		# print("################################################")
		# candidate_solution = np.array([-0.11979351, -0.09333089,  0.14412426,  0.13925383,  0.09697775,  0.03665841,
		# 	0.38079478,  0.31782679, -0.09528776, -0.02311008])
		# candidate_solution = np.array([-0.07964954,  0.09430439,  0.13646698,  0.09961439,  0.08887494, -0.00242865,
  # 0.34886347,  0.24144642, -0.221433,   -0.02403533,])
		print("Candidate solution:")
		print(candidate_solution)
		NSF = False
		if type(candidate_solution) == str and candidate_solution == 'NSF':
			passed_safety=False
			NSF = True
		else:
			# Safety test
			st_kwargs = dict(dataset=safety_dataset,
				model=model,
				parse_trees=parse_trees,
				regime=regime
				)

			if regime == 'RL':
				st_kwargs['gamma'] = RL_environment_obj.gamma
				st_kwargs['min_return'] = RL_environment_obj.min_return
				st_kwargs['max_return'] = RL_environment_obj.max_return

			st = SafetyTest(**st_kwargs)

			passed_safety = st.run(candidate_solution,
				bound_method='ttest')

		failed=False # we said we passed the safety test but
		# the constraint fails on the original dataset (ground truth)

		if NSF:
			print("NSF")
			performance = -99.0
		else:
			# If passed the safety test, calculate performance
			# using candidate solution 
			if passed_safety:
				print("Passed safety test. Calculating performance")
				if regime == 'supervised':
					print("Calculating performance with candidate solution:")
					print(candidate_solution)
					
					prediction_test = model.predict(
						candidate_solution,
						test_features)
					predict_class = prediction_test>=0.5
					acc = np.mean(1.0*predict_class==test_labels)
					performance = acc
					print(f"Accuracy = {performance}")
				
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
		
	
				print("Determining whether solution is actually safe on ground truth")
				
				
				if regime == 'supervised':
					for parse_tree in parse_trees:
						parse_tree.evaluate_constraint(theta=candidate_solution,
							dataset=dataset,
							model=model,regime='supervised',
							branch='safety_test')
						ghat = parse_tree.root.value
						if ghat > 0:
							failed = True

				elif regime == 'RL':
					ghat = gfunc(
						df_regen,
						RL_environment_obj=RL_environment_obj,
						)
					if ghat > 0:
						failed = True
				print(f"ghat on test set: {ghat}")
				
				if failed:
					print("Solution was not actually safe on test set!")
				else:
					print("Solution was safe on test set")
				print()
			else:
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
