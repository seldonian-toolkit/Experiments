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

		test_features = kwargs['test_features']	
		if kwargs['include_intercept_col']:
			test_features.insert(0,'offset',1.0) # inserts a column of 1's
		test_labels = kwargs['test_labels']

		helper = partial(
			self.QSA,
			test_features=test_features,
			test_labels=test_labels,
			dataset=kwargs['dataset'],
			parse_trees=kwargs['parse_trees'],
			constraint_funcs=kwargs['constraint_funcs'],
			frac_data_in_safety=kwargs['frac_data_in_safety'],
			eval_method=kwargs['eval_method'],
			precalc_dict=precalc_dict,
			results_dir=kwargs['results_dir'],
			max_iter=kwargs['max_iter'],
			n_jobs=kwargs['n_jobs'],
			seldonian_model_type=kwargs['seldonian_model_type'],
			optimizer=kwargs['optimizer'],
			verbose=kwargs['verbose'],
		)
		data_pcts = kwargs['data_pcts']
		n_trials = kwargs['n_trials']
		data_pcts_vector = np.array([x for x in data_pcts for y in range(n_trials)])
		trials_vector = np.array([x for y in range(len(data_pcts)) for x in range(n_trials)])

		with ProcessPoolExecutor(max_workers=kwargs['n_jobs']) as ex:
			results = tqdm(ex.map(helper,data_pcts_vector,trials_vector),total=len(data_pcts_vector))
			for exc in results:
				if exc:
					print(exc)
		self._aggregate_results(**kwargs)
	
	def QSA(self,data_pct,trial_i,**kwargs):
		try: 
			verbose=kwargs['verbose']
			trial_dir = os.path.join(
					kwargs['results_dir'],
					'qsa_results',
					'trial_data')

			os.makedirs(trial_dir,exist_ok=True)
			
			parse_trees = kwargs['parse_trees']
			dataset = kwargs['dataset']
			orig_df = dataset.df
			# number of points in this partition of the data
			n_points = int(round(data_pct*len(orig_df))) 

			test_features = kwargs['test_features']
			test_labels = kwargs['test_labels']
			sensitive_column_names = dataset.sensitive_column_names
			label_column = dataset.label_column

			if kwargs['eval_method'] == 'resample':
				df = orig_df.sample(n=len(orig_df),replace=True)[0:n_points]

			
			candidate_df, safety_df = train_test_split(
				df, test_size=kwargs['frac_data_in_safety'], shuffle=False)
	
			sensitive_candidate_df = candidate_df[sensitive_column_names]
			sensitive_safety_df = safety_df[sensitive_column_names]
			
			candidate_dataset = DataSet(
				candidate_df,meta_information=df.columns,
				sensitive_column_names=sensitive_column_names,
				regime='supervised',label_column=label_column)

			safety_dataset = DataSet(
				safety_df,meta_information=df.columns,
				sensitive_column_names=sensitive_column_names,
				regime='supervised',label_column=label_column)
			
			n_safety = len(safety_df)

			# Linear regression model for classification
			if kwargs['seldonian_model_type'] == 'linear_classifier':
				model = LinearClassifierModel()

			# Candidate selection
			minimizer_options = {}
			
			cs = CandidateSelection(
				model=model,
				candidate_dataset=candidate_dataset,
				n_safety=n_safety,
				parse_trees=parse_trees,
				primary_objective=model.perceptron_loss,
				optimization_technique='barrier_function',
				optimizer=kwargs['optimizer'],
				initial_solution_fn=model.fit)

			candidate_solution = cs.run(minimizer_options=minimizer_options)
			
			# Safety test
			st = SafetyTest(safety_dataset,model,parse_trees)
			passed_safety = st.run(candidate_solution,
				bound_method='ttest')

			# Calculate accuracy from entire original dataset 

			prediction = model.predict(
				candidate_solution,
				test_features)

			acc = np.mean(1.0*prediction==test_labels)
			# Calculate whether we failed, i.e. 
			# we said we passed the safety test but
			# the constraint fails on the original dataset (ground truth)
			failed=False
			if passed_safety:
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
				passed_safety,
				failed]
			colnames = ['data_pct','trial_i','acc','passed_safety','failed']
			self.write_trial_result(
				data,
				colnames,
				trial_dir,
				verbose=kwargs['verbose'])

		except Exception as e:
			return e
		return None
