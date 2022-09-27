""" Module for running Seldonian Experiments """

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
from sklearn.linear_model import LogisticRegression

from seldonian.utils.io_utils import load_pickle
from seldonian.dataset import (SupervisedDataSet,RLDataSet)
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.models.models import (LogisticRegressionModel,
	DummyClassifierModel,RandomClassifierModel)

from fairlearn.reductions import ExponentiatedGradient
from fairlearn.metrics import (MetricFrame,selection_rate,
	false_positive_rate,true_positive_rate,false_negative_rate)
from fairlearn.reductions import (
	DemographicParity,FalsePositiveRateParity,
	EqualizedOdds)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

class Experiment():
	def __init__(self,model_name,results_dir):
		""" Base class for running experiments

		:param model_name: The string name of the baseline model, 
			e.g 'logistic_regression'
		:type model_name: str

		:param results_dir: Parent directory for saving any
			experimental results
		:type results_dir: str

		"""
		self.model_name = model_name
		self.results_dir = results_dir

	def aggregate_results(self,**kwargs):
		""" Group together the data in each 
		trial file into a single CSV file.
		"""
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
		for data_frac in kwargs['data_fracs']:
			for trial_i in range(kwargs['n_trials']):
				filename = os.path.join(trial_dir,
					f'data_frac_{data_frac:.4f}_trial_{trial_i}.csv')
				df = pd.read_csv(filename)
				df_list.append(df)

		result_df = pd.concat(df_list)
		result_df.to_csv(savename_results,index=False)
		print(f"Saved {savename_results}")
		return

	def write_trial_result(self,data,
		colnames,trial_dir, 
		verbose=False):
		""" Write out the results from a single trial
		to a file.

		:param data: The information to save
		:type data: List

		:param colnames: Names of the items in the list. 
			These will comprise the header of the saved file
		:type colnames: List(str)

		:param trial_dir: The directory in which to save the file
		:type trial_dir: str

		:param verbose: if True, prints out saved filename
		:type verbose: bool
		"""
		result_df = pd.DataFrame([data])
		result_df.columns = colnames
		data_frac,trial_i = data[0:2]

		savename = os.path.join(trial_dir,
			f'data_frac_{data_frac:.4f}_trial_{trial_i}.csv')

		result_df.to_csv(savename,index=False)
		if verbose:
			print(f"Saved {savename}")
		return

class BaselineExperiment(Experiment):
	def __init__(self,model_name,results_dir):
		""" Class for running baseline experiments
		against which to compare Seldonian Experiments 

		:param model_name: The string name of the baseline model, 
			e.g 'logistic_regression'
		:type model_name: str

		:param results_dir: Parent directory for saving any
			experimental results
		:type results_dir: str
		"""
		super().__init__(model_name,results_dir)

	def run_experiment(self,**kwargs):
		""" Run the baseline experiment """

		helper = partial(
			self.run_baseline_trial,
			spec=kwargs['spec'],
			datagen_method=kwargs['datagen_method'],
			perf_eval_fn=kwargs['perf_eval_fn'],
			perf_eval_kwargs=kwargs['perf_eval_kwargs'],
			verbose=kwargs['verbose'],
			n_workers=kwargs['n_workers'],
		)
		data_fracs = kwargs['data_fracs']
		n_trials = kwargs['n_trials']
		n_workers=kwargs['n_workers']

		data_fracs_vector = np.array([x for x in data_fracs for y in range(n_trials)])
		trials_vector = np.array([x for y in range(len(data_fracs)) for x in range(n_trials)])
		
		if n_workers == 1:
			for ii in range(len(data_fracs_vector)):
				data_frac = data_fracs_vector[ii]
				trial_i = trials_vector[ii]
				print(data_frac,trial_i)
				helper(data_frac,trial_i)
		elif n_workers > 1:
			with ProcessPoolExecutor(max_workers=n_workers,
				mp_context=mp.get_context('fork')) as ex:
				results = tqdm(ex.map(helper,data_fracs_vector,trials_vector),
					total=len(data_fracs_vector))
				for exc in results:
					if exc:
						print(exc)
		else:
			raise ValueError(f"value of {n_workers} must be >=1 ")

		self.aggregate_results(**kwargs)
	
	def run_baseline_trial(self,data_frac,trial_i,**kwargs):
		""" Run a trial of the baseline model  
		
		:param data_frac: Fraction of overall dataset size to use
		:type data_frac: float

		:param trial_i: The index of the trial 
		:type trial_i: int
		"""

		spec = kwargs['spec']
		dataset = spec.dataset
		parse_trees = spec.parse_trees
		verbose=kwargs['verbose']
		datagen_method = kwargs['datagen_method']
		perf_eval_fn = kwargs['perf_eval_fn']
		perf_eval_kwargs = kwargs['perf_eval_kwargs']

		trial_dir = os.path.join(
				self.results_dir,
				f'{self.model_name}_results',
				'trial_data')

		os.makedirs(trial_dir,exist_ok=True)

		savename = os.path.join(trial_dir,
			f'data_frac_{data_frac:.4f}_trial_{trial_i}.csv')

		if os.path.exists(savename):
			if verbose:
				print(f"Trial {trial_i} already run for "
					  f"this data_frac: {data_frac}. Skipping this trial. ")
			return

		##############################################
		""" Setup for running baseline algorithm """
		##############################################

		sensitive_column_names = dataset.sensitive_column_names
		include_sensitive_columns = dataset.include_sensitive_columns
		include_intercept_term = dataset.include_intercept_term
		label_column = dataset.label_column

		if datagen_method == 'resample':
			resampled_filename = os.path.join(self.results_dir,
				'resampled_dataframes',f'trial_{trial_i}.pkl')
			n_points = int(round(data_frac*len(dataset.df))) 

			resampled_df = load_pickle(resampled_filename).iloc[:n_points]

			if verbose:
				print(f"Using resampled dataset {resampled_filename} "
					  f"with {len(resampled_df)} datapoints")
		else:
			raise NotImplementedError(
				f"datagen_method: {datagen_method} "
				f"not supported for regime: {regime}")
		
		# Prepare features and labels

		features = resampled_df.loc[:,
		    resampled_df.columns != label_column]
		labels = resampled_df[label_column].astype('int')

		# Drop sensitive features from training set
		if not include_sensitive_columns:
		    features = features.drop(
		        columns=dataset.sensitive_column_names)	

		if dataset.include_intercept_term:
			features.insert(0,'offset',1.0) # inserts a column of 1's
		
		####################################################
		"""" Instantiate model and fit to resampled data """
		####################################################
		# X_test_baseline = perf_eval_kwargs['X'].drop(columns=['offset'])
		X_test_baseline = perf_eval_kwargs['X']
		
		if self.model_name == 'logistic_regression':
			baseline_model = LogisticRegressionModel()
			try:
				solution = baseline_model.fit(features, labels)
				# predict the probabilities not the class labels
				y_pred = baseline_model.predict(solution,X_test_baseline)
			except ValueError:
				solution = "NSF"
		
		elif self.model_name == 'random_classifier':
			# Returns the positive class with p=0.5 every time
			baseline_model = RandomClassifierModel()
			solution = None
			y_pred = baseline_model.predict(solution,X_test_baseline)
		
		#########################################################
		"""" Calculate performance and safety on ground truth """
		#########################################################
		# Handle whether solution was found 
		solution_found=True
		if type(solution) == str and solution == 'NSF':
			solution_found = False
		
		#########################################################
		"""" Calculate performance and safety on ground truth """
		#########################################################
		
		failed=False # flag for whether we were actually safe on test set
		if solution_found:
			performance = perf_eval_fn(
				y_pred,
				**perf_eval_kwargs)

			if verbose:
				print(f"Performance = {performance}")

			# Determine whether this solution
			# violates any of the constraints 
			# on the test dataset
			for parse_tree in parse_trees:
				parse_tree.reset_base_node_dict(reset_data=True)
				parse_tree.evaluate_constraint(theta=solution,
					dataset=dataset,
					model=baseline_model,regime='supervised_learning',
					branch='safety_test')

				g = parse_tree.root.value
				print(f"g (logistic regression) = {g}")
				if g > 0 or np.isnan(g):
					failed = True
					if verbose:
						print("Failed on test set")
				if verbose:
					print(f"g = {g}")
		else:
			print("NSF")
			performance = np.nan

		# Write out file for this data_frac,trial_i combo
		data = [data_frac,
			trial_i,
			performance,
			failed]
		colnames = ['data_frac','trial_i','performance','failed']
		self.write_trial_result(
			data,
			colnames,
			trial_dir,
			verbose=kwargs['verbose'])
		return 

class FairlearnExperiment(Experiment):

	def __init__(self,results_dir,fairlearn_epsilon_constraint):
		""" Class for running Fairlearn experiments

		:param results_dir: Parent directory for saving any
			experimental results
		:type results_dir: str

		:param fairlearn_epsilon_constraint: The value of epsilon
			(the threshold) to use in the constraint 
			to the Fairlearn model
		:type fairlearn_epsilon_constraint: float
		"""
		super().__init__(results_dir=results_dir,
			model_name=f'fairlearn_eps{fairlearn_epsilon_constraint:.2f}')

	def run_experiment(self,**kwargs):
		""" Run the Fairlearn experiment """
		n_workers = kwargs['n_workers']
		partial_kwargs = {key:kwargs[key] for key in kwargs \
			if key not in ['data_fracs','n_trials']}

		helper = partial(
			self.run_fairlearn_trial,
			**partial_kwargs
		)

		data_fracs = kwargs['data_fracs']
		n_trials = kwargs['n_trials']
		data_fracs_vector = np.array([x for x in data_fracs for y in range(n_trials)])
		trials_vector = np.array([x for y in range(len(data_fracs)) for x in range(n_trials)])

		if n_workers == 1:
			for ii in range(len(data_fracs_vector)):
				data_frac = data_fracs_vector[ii]
				trial_i = trials_vector[ii]
				print(data_frac,trial_i)
				helper(data_frac,trial_i)
		elif n_workers > 1:
			with ProcessPoolExecutor(max_workers=n_workers,
				mp_context=mp.get_context('fork')) as ex:
				results = tqdm(ex.map(helper,data_fracs_vector,trials_vector),
					total=len(data_fracs_vector))
				for exc in results:
					if exc:
						print(exc)
		else:
			raise ValueError(f"n_workers value of {n_workers} must be >=1 ")

		self.aggregate_results(**kwargs)
	
	def run_fairlearn_trial(self,data_frac,trial_i,**kwargs):
		""" Run a Fairlearn trial 
		
		:param data_frac: Fraction of overall dataset size to use
		:type data_frac: float

		:param trial_i: The index of the trial 
		:type trial_i: int
		"""
		spec = kwargs['spec']
		verbose=kwargs['verbose']
		datagen_method = kwargs['datagen_method']
		fairlearn_sensitive_feature_names = kwargs['fairlearn_sensitive_feature_names']
		fairlearn_constraint_name = kwargs['fairlearn_constraint_name']
		fairlearn_epsilon_constraint = kwargs['fairlearn_epsilon_constraint']
		fairlearn_epsilon_eval = kwargs['fairlearn_epsilon_eval']
		fairlearn_eval_kwargs = kwargs['fairlearn_eval_kwargs']
		perf_eval_fn = kwargs['perf_eval_fn']
		perf_eval_kwargs = kwargs['perf_eval_kwargs']
		constraint_eval_fns = kwargs['constraint_eval_fns']
		constraint_eval_kwargs = kwargs['constraint_eval_kwargs']

		regime=spec.dataset.regime
		assert regime == 'supervised_learning'

		trial_dir = os.path.join(
				self.results_dir,
				f'fairlearn_eps{fairlearn_epsilon_constraint:.2f}_results',
				'trial_data')

		savename = os.path.join(trial_dir,
			f'data_frac_{data_frac:.4f}_trial_{trial_i}.csv')

		if os.path.exists(savename):
			if verbose:
				print(f"Trial {trial_i} already run for "
					  f"this data_frac: {data_frac}. Skipping this trial. ")
			return

		os.makedirs(trial_dir,exist_ok=True)
		
		parse_trees = spec.parse_trees
		dataset = spec.dataset

		##############################################
		""" Setup for running Fairlearn algorithm """
		##############################################

		sensitive_column_names = dataset.sensitive_column_names
		include_sensitive_columns = dataset.include_sensitive_columns
		include_intercept_term = False 
		label_column = dataset.label_column

		if datagen_method == 'resample':
			resampled_filename = os.path.join(self.results_dir,
				'resampled_dataframes',f'trial_{trial_i}.pkl')
			n_points = int(round(data_frac*len(dataset.df))) 

			with open(resampled_filename,'rb') as infile:
				resampled_df = pickle.load(infile).iloc[:n_points]

			if verbose:
				print(f"Using resampled dataset {resampled_filename} "
					  f"with {len(resampled_df)} datapoints")
		else:
			raise NotImplementedError(
				f"datagen_method: {datagen_method} "
				f"not supported for regime: {regime}")
		

		# Prepare features and labels

		features = resampled_df.loc[:,
		    resampled_df.columns != label_column]
		labels = resampled_df[label_column].astype('int')

		# Drop sensitive features from training set
		if not include_sensitive_columns:
		    features = features.drop(
		        columns=dataset.sensitive_column_names)	

		# Get fairlearn sensitive features
		fairlearn_sensitive_features = resampled_df.loc[:,
			fairlearn_sensitive_feature_names]

		##############################################
		"""" Run Fairlearn algorithm on trial data """
		##############################################

		if fairlearn_constraint_name == "disparate_impact":
			fairlearn_constraint = DemographicParity(
				ratio_bound=fairlearn_epsilon_constraint)
		
		elif fairlearn_constraint_name == "demographic_parity":
			fairlearn_constraint = DemographicParity(
				difference_bound=fairlearn_epsilon_constraint)
		
		elif fairlearn_constraint_name == "predictive_equality":
			fairlearn_constraint = FalsePositiveRateParity(
				difference_bound=fairlearn_epsilon_constraint)
		
		elif fairlearn_constraint_name == "equalized_odds":
			fairlearn_constraint = EqualizedOdds(
				difference_bound=fairlearn_epsilon_constraint)

		elif fairlearn_constraint_name == "equal_opportunity":
			fairlearn_constraint = EqualizedOdds(
				difference_bound=fairlearn_epsilon_constraint)
		
		else:
			raise NotImplementedError(
				"Fairlearn constraints of type: "
			   f"{fairlearn_constraint_name} "
			    "is not supported.")

		classifier = LogisticRegression()
		
		mitigator = ExponentiatedGradient(classifier, 
			fairlearn_constraint)
		solution_found = True

		try:
			mitigator.fit(features, labels, 
				sensitive_features=fairlearn_sensitive_features)
			X_test_fairlearn = fairlearn_eval_kwargs['X'] # same as X_test but drops the offset column
			y_pred = self.get_fairlearn_predictions(mitigator,X_test_fairlearn)
		except:
			print("Error when fitting. Returning NSF")
			solution_found = False
			performance = np.nan
		#########################################################
		"""" Calculate performance and safety on ground truth """
		#########################################################
		if solution_found:
			fairlearn_eval_kwargs['model'] = mitigator

			performance = perf_eval_fn(
				y_pred,
				**fairlearn_eval_kwargs)
				
		if verbose:
			print(f"Performance = {performance}")
			
		########################################
		""" Calculate safety on ground truth """
		########################################
		if verbose:
			print("Determining whether solution "
				  "is actually safe on ground truth")
		
		# predict the class label, not the probability
		# Determine whether this solution
		# violates any of the constraints 
		# on the test dataset
		failed=False
		if solution_found:
			fairlearn_eval_method = fairlearn_eval_kwargs['eval_method']
			failed = self.evaluate_constraint_function(
				y_pred=y_pred,
				test_labels=fairlearn_eval_kwargs['y'],
				fairlearn_constraint_name=fairlearn_constraint_name,
				epsilon_eval=fairlearn_epsilon_eval,
				eval_method=fairlearn_eval_method,
				sensitive_features=fairlearn_eval_kwargs['sensitive_features'])
		else:
			failed = False

		if failed:
			print("Fairlearn trial UNSAFE on ground truth")
		else:
			print("Fairlearn trial SAFE on ground truth")
		# Write out file for this data_frac,trial_i combo
		data = [data_frac,
			trial_i,
			performance,
			failed]

		colnames = ['data_frac','trial_i','performance','failed']
		self.write_trial_result(
			data,
			colnames,
			trial_dir,
			verbose=kwargs['verbose'])
		return
	
	def get_fairlearn_predictions(self,mitigator,X_test_fairlearn):
		"""
		Get the predicted labels from the fairlearn mitigator.
		The mitigator consists of potentially more than one predictor
		and as many weights as predictors. For each predictor with non-zero
		weight, predict the proportion of points given by the weight
		of each predictor. 

		:param mitigator: The Fairlearn mitigator

		:param X_test_fairlearn: The test features from which 
			to predict the labels
		"""
		y_pred = np.zeros(len(X_test_fairlearn))
		assert len(mitigator.predictors_) == len(mitigator.weights_)
		start_index = 0
		for ii in range(len(mitigator.predictors_)):
			weight = mitigator.weights_[ii]
			if weight == 0:
				continue
			predictor = mitigator.predictors_[ii]
			n_points_this_predictor = int(round(weight*len(X_test_fairlearn)))
			end_index = start_index + n_points_this_predictor
			X_test_this_predictor = X_test_fairlearn.iloc[start_index:end_index]

			probs = predictor.predict_proba(X_test_this_predictor)
			predictions = predictor.predict_proba(X_test_this_predictor)[:,1]
			y_pred[start_index:end_index] = predictions
			start_index = end_index
		return y_pred
	
	def evaluate_constraint_function(self,
		y_pred,
		test_labels,
		fairlearn_constraint_name,
		epsilon_eval,
		eval_method='native',
		sensitive_features=[]):
		""" Evaluate the constraint function using the 
		Fairlearn predictions

		:param y_pred: Predicted labels, same shape as test_labels
		:type y_pred: 1D array

		:param test_labels: True labels
		:type test_labels: 1D array

		:param fairlearn_constraint_name: The name of the constraint
		:type fairlearn_constraint_name: str

		:param epsilon_eval: The threshold in the constraint to use for 
			evaluation
		:type epsilon_eval: float

		:param eval_method: The method for evaluating the constraint, 
			two options: 'native' or 'two-groups'
		:type eval_method: str, defaults to 'native'
		
		:param sensitive_features: List of column names that are considered
			sensitive in the Fairlearn dataset
		:type sensitive_features: List(str)
		"""
		print(f"Evaluating constraint for: {fairlearn_constraint_name}")
		failed = False

		if fairlearn_constraint_name == "demographic_parity":
			# g = abs((PR | ATR1) - (PR | ATR2)) - eps
			PR_frame = MetricFrame(
				metrics=selection_rate,
				y_true=test_labels, y_pred=y_pred>=0.5,
				sensitive_features=sensitive_features)
			PR_grouped = PR_frame.by_group
			PR_group1 = PR_grouped.iloc[0]
			if eval_method == 'native':
				PR_overall = PR_frame.overall
				g = abs(PR_group1 - PR_overall) - epsilon_eval
			elif eval_method == 'two-groups':
				PR_group2 = PR_grouped.iloc[1]
				g = abs(PR_group1 - PR_group2) - epsilon_eval

		elif fairlearn_constraint_name == "predictive_equality":
			# g = abs((FPR | ATR1) - (FPR | ATR2)) - eps
			FPR_frame = MetricFrame(
				metrics=false_positive_rate,
				y_true=test_labels, y_pred=y_pred>=0.5,
				sensitive_features=sensitive_features)
			FPR_grouped = FPR_frame.by_group
			FPR_group1 = FPR_grouped.iloc[0]
			if eval_method == 'native':
				FPR_overall = FPR_frame.overall
				g = abs(FPR_group1 - FPR_overall) - epsilon_eval
			elif eval_method == 'two-groups':
				FPR_group2 = FPR_grouped.iloc[1]
				g = abs(FPR_group1 - FPR_group2) - epsilon_eval

		elif fairlearn_constraint_name == "disparate_impact":
			# g = epsilon - min((PR | ATR1)/(PR | ATR2),(PR | ATR2)/(PR | ATR1))
			PR_frame = MetricFrame(
				metrics=selection_rate,
				y_true=test_labels, y_pred=y_pred>=0.5,
				sensitive_features=sensitive_features)

			PR_grouped = PR_frame.by_group
			PR_group1 = PR_grouped.iloc[0]
			if eval_method == 'native':
				PR_overall = PR_frame.overall
				g = epsilon_eval - min(PR_group1/PR_overall,PR_overall/PR_group1) 
			elif eval_method == 'two-groups':
				PR_group2 = PR_grouped.iloc[1]
				g = epsilon_eval - min(PR_group1/PR_group2,PR_group2/PR_group1) 

		elif fairlearn_constraint_name == "equalized_odds":
			# g = abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) - epsilon
			FPR_frame = MetricFrame(
				metrics=false_positive_rate,
				y_true=test_labels, y_pred=y_pred>=0.5,
				sensitive_features=sensitive_features)
			FPR_grouped = FPR_frame.by_group
			FPR_group1 = FPR_grouped.iloc[0]

			FNR_frame = MetricFrame(
				metrics=false_negative_rate,
				y_true=test_labels, y_pred=y_pred>=0.5,
				sensitive_features=sensitive_features)
			FNR_grouped = FNR_frame.by_group
			FNR_group1 = FNR_grouped.iloc[0]
			
			if eval_method == 'native':
				FPR_overall = FPR_frame.overall
				FNR_overall = FNR_frame.overall	
				g = abs(FPR_group1 - FPR_overall) + abs(FNR_group1 - FNR_overall) - epsilon_eval
			elif eval_method == 'two-groups':
				FPR_group2 = FPR_grouped.iloc[1]
				FNR_group2 = FNR_grouped.iloc[1]
				g = abs(FPR_group1 - FPR_group2) + abs(FNR_group1 - FNR_group2) - epsilon_eval
		
		elif fairlearn_constraint_name == "equal_opportunity":
			# g = abs((FNR | [M]) - (FNR | [F])) - epsilon
			
			FNR_frame = MetricFrame(
				metrics=false_negative_rate,
				y_true=test_labels, y_pred=y_pred>=0.5,
				sensitive_features=sensitive_features)
			FNR_grouped = FNR_frame.by_group
			FNR_group1 = FNR_grouped.iloc[0]
			
			if eval_method == 'native':
				FNR_overall = FNR_frame.overall	
				g = abs(FNR_group1 - FNR_overall) - epsilon_eval
			elif eval_method == 'two-groups':
				FNR_group2 = FNR_grouped.iloc[1]
				g = abs(FNR_group1 - FNR_group2) - epsilon_eval
		else:
			raise NotImplementedError(
				"Evaluation for Fairlearn constraints of type: "
			   f"{fairlearn_constraint.short_name} "
			    "is not supported.")
		
		print(f"g (fairlearn) = {g}")
		if g > 0 or np.isnan(g):
			failed = True

		return failed

class SeldonianExperiment(Experiment):
	def __init__(self,model_name,results_dir):
		""" Class for running Seldonian experiments

		:param model_name: The string name of the Seldonian model, 
			only option is currently: 'qsa' (quasi-Seldonian algorithm) 
		:type model_name: str

		:param results_dir: Parent directory for saving any
			experimental results
		:type results_dir: str

		"""
		super().__init__(model_name,results_dir)
		if self.model_name != 'qsa':
			raise NotImplementedError(
				"Seldonian experiments for model: "
				f"{self.model_name} are not supported.")

	def run_experiment(self,**kwargs):
		""" Run the Seldonian experiment """
		n_workers = kwargs['n_workers']
		partial_kwargs = {key:kwargs[key] for key in kwargs \
			if key not in ['data_fracs','n_trials']}

		# Pass partial_kwargs onto self.QSA()
		helper = partial(
			self.QSA,
			**partial_kwargs
		)

		data_fracs = kwargs['data_fracs']
		n_trials = kwargs['n_trials']
		data_fracs_vector = np.array([x for x in data_fracs for y in range(n_trials)])
		trials_vector = np.array([x for y in range(len(data_fracs)) for x in range(n_trials)])

		if n_workers == 1:
			for ii in range(len(data_fracs_vector)):
				data_frac = data_fracs_vector[ii]
				trial_i = trials_vector[ii]
				print(data_frac,trial_i)
				helper(data_frac,trial_i)
		elif n_workers > 1:
			with ProcessPoolExecutor(max_workers=n_workers,
				mp_context=mp.get_context('fork')) as ex:
				results = tqdm(ex.map(helper,data_fracs_vector,trials_vector),
					total=len(data_fracs_vector))
				for exc in results:
					if exc:
						print(exc)
		else:
			raise ValueError(f"n_workers value of {n_workers} must be >=1 ")

		self.aggregate_results(**kwargs)
	
	def QSA(self,data_frac,trial_i,**kwargs):
		""" Run a trial of the quasi-Seldonian algorithm  
		
		:param data_frac: Fraction of overall dataset size to use
		:type data_frac: float

		:param trial_i: The index of the trial 
		:type trial_i: int
		"""
		spec = kwargs['spec']
		verbose=kwargs['verbose']
		datagen_method = kwargs['datagen_method']
		perf_eval_fn = kwargs['perf_eval_fn']
		perf_eval_kwargs = kwargs['perf_eval_kwargs']
		constraint_eval_fns = kwargs['constraint_eval_fns']
		constraint_eval_kwargs = kwargs['constraint_eval_kwargs']

		regime=spec.dataset.regime

		trial_dir = os.path.join(
				self.results_dir,
				'qsa_results',
				'trial_data')

		savename = os.path.join(trial_dir,
			f'data_frac_{data_frac:.4f}_trial_{trial_i}.csv')

		if os.path.exists(savename):
			if verbose:
				print(f"Trial {trial_i} already run for "
					  f"this data_frac: {data_frac}. Skipping this trial. ")
			return

		os.makedirs(trial_dir,exist_ok=True)
		
		parse_trees = spec.parse_trees
		dataset = spec.dataset

		##############################################
		""" Setup for running Seldonian algorithm """
		##############################################

		if regime == 'supervised_learning':

			sensitive_column_names = dataset.sensitive_column_names
			include_sensitive_columns = dataset.include_sensitive_columns
			include_intercept_term = dataset.include_intercept_term
			label_column = dataset.label_column

			if datagen_method == 'resample':
				resampled_filename = os.path.join(self.results_dir,
					'resampled_dataframes',f'trial_{trial_i}.pkl')
				n_points = int(round(data_frac*len(dataset.df))) 
				if n_points < 1:
					raise ValueError(
						f"This data_frac={data_frac} "
						f"results in {n_points} data points. "
						 "Must have at least 1 data point to run a trial.")
				with open(resampled_filename,'rb') as infile:
					resampled_df = pickle.load(infile).iloc[:n_points]

				if verbose:
					print(f"Using resampled dataset {resampled_filename} "
						  f"with {len(resampled_df)} datapoints")
			else:
				raise NotImplementedError(
					f"Eval method {datagen_method} "
					f"not supported for regime={regime}")
			dataset_for_experiment = SupervisedDataSet(
				df=resampled_df,
				meta_information=resampled_df.columns,
				label_column=label_column,
				sensitive_column_names=sensitive_column_names,
				include_sensitive_columns=include_sensitive_columns,
				include_intercept_term=include_intercept_term)

			# Make a new spec object where the 
			# only thing that is different is the dataset

			spec_for_experiment = copy.deepcopy(spec)
			spec_for_experiment.dataset = dataset_for_experiment

		elif regime == 'reinforcement_learning':
			hyperparameter_and_setting_dict = kwargs['hyperparameter_and_setting_dict']
			
			if datagen_method == 'generate_episodes':
				n_episodes_for_eval = perf_eval_kwargs['n_episodes_for_eval']
				# Sample from resampled dataset on disk of n_episodes
				save_dir = os.path.join(self.results_dir,'regenerated_datasets')

				savename = os.path.join(save_dir,f'regenerated_data_trial{trial_i}.pkl')
				
				episodes_all = load_pickle(savename)
				# Take data_frac episodes from this df
				n_episodes_all = len(episodes_all)

				n_episodes_for_exp = int(round(n_episodes_all*data_frac))
				if n_episodes_for_exp < 1:
					raise ValueError(
						f"This data_frac={data_frac} "
						f"results in {n_episodes_for_exp} episodes. "
						 "Must have at least 1 episode to run a trial.")

				print(f"Orig dataset should have {n_episodes_all} episodes")
				print(f"This dataset with data_frac={data_frac} should have"
					f" {n_episodes_for_exp} episodes")
				
				# Take first n_episodes episodes 
				episodes_for_exp = episodes_all[0:n_episodes_for_exp]
				assert len(episodes_for_exp) == n_episodes_for_exp

				dataset_for_experiment = RLDataSet(
					episodes=episodes_for_exp,
					meta_information=dataset.meta_information,
					regime=regime)

				# Make a new spec object from a copy of spec, where the 
				# only thing that is different is the dataset

				spec_for_experiment = copy.deepcopy(spec)
				spec_for_experiment.dataset = dataset_for_experiment
			else:
				raise NotImplementedError(
					f"Eval method {datagen_method} "
					"not supported for regime={regime}")
		
		################################
		"""" Run Seldonian algorithm """
		################################

		try:
			SA = SeldonianAlgorithm(
				spec_for_experiment)
			passed_safety,solution = SA.run(write_cs_logfile=True)
			
		except (ValueError,ZeroDivisionError):
			passed_safety=False
			solution = "NSF"

		print("Solution from running seldonian algorithm:")
		print(solution)
		print()
		
		# Handle whether solution was found 
		solution_found=True
		if type(solution) == str and solution == 'NSF':
			solution_found = False
		
		#########################################################
		"""" Calculate performance and safety on ground truth """
		#########################################################
		
		failed=False # flag for whether we were actually safe on test set

		if solution_found:
			solution = copy.deepcopy(solution)
			# If passed the safety test, calculate performance
			# using solution 
			if passed_safety:
				if verbose:
					print("Passed safety test! Calculating performance")

				#############################
				""" Calculate performance """
				#############################
				if regime == 'supervised_learning':
					X_test = perf_eval_kwargs['X']
					model = copy.deepcopy(spec_for_experiment.model)
					y_pred = model.predict(solution,X_test)
					performance = perf_eval_fn(
						y_pred,
						**perf_eval_kwargs)
				
				if regime == 'reinforcement_learning':
					model = copy.deepcopy(SA.model)
					model.policy.set_new_params(solution)
					perf_eval_kwargs['model'] = model
					perf_eval_kwargs['hyperparameter_and_setting_dict'] = hyperparameter_and_setting_dict
					episodes_for_eval,performance = perf_eval_fn(**perf_eval_kwargs)
				if verbose:
					print(f"Performance = {performance}")
			
				########################################
				""" Calculate safety on ground truth """
				########################################
				if verbose:
					print("Determining whether solution "
						  "is actually safe on ground truth")
				
				if constraint_eval_fns == []:
					constraint_eval_kwargs['model']=model
					constraint_eval_kwargs['spec_orig']=spec
					constraint_eval_kwargs['spec_for_experiment']=spec_for_experiment
					constraint_eval_kwargs['regime']=regime
					constraint_eval_kwargs['branch'] = 'safety_test'
					constraint_eval_kwargs['verbose']=verbose

				if regime == 'reinforcement_learning':
					constraint_eval_kwargs['episodes_for_eval'] = episodes_for_eval

				failed = self.evaluate_constraint_functions(
					solution=solution,
					constraint_eval_fns=constraint_eval_fns,
					constraint_eval_kwargs=constraint_eval_kwargs)
				
				if verbose:
					if failed:
						print("Solution was not actually safe on ground truth!")
					else:
						print("Solution was safe on ground truth")
					print()
			else:
				if verbose:
					print("Failed safety test ")
					performance = np.nan
		
		else:
			print("NSF")
			performance = np.nan
		# Write out file for this data_frac,trial_i combo
		data = [data_frac,
			trial_i,
			performance,
			passed_safety,
			failed]
		colnames = ['data_frac','trial_i','performance','passed_safety','failed']
		self.write_trial_result(
			data,
			colnames,
			trial_dir,
			verbose=kwargs['verbose'])
		return

	def evaluate_constraint_functions(self,
		solution,constraint_eval_fns,
		constraint_eval_kwargs):
		""" Helper function for QSA() to evaluate
		the constraint functions to determine
		whether solution was safe on ground truth

		:param solution: The weights of the model found
			during candidate selection in a given trial
		:type solution: numpy ndarray

		:param constraint_eval_fns: List of functions
			to use to evaluate each constraint. 
			An empty list results in using the parse
			tree to evaluate the constraints
		:type constraint_eval_fns: List(function)

		:param constraint_eval_kwargs: keyword arguments
			to pass to each constraint function 
			in constraint_eval_fns
		:type constraint_eval_kwargs: dict
		"""
		# Use safety test branch so the confidence bounds on
		# leaf nodes are not inflated
		failed = False
		if constraint_eval_fns == []:
			""" User did not provide their own functions
			to evaluate the constraints. Use the default: 
			the parse tree has a built-in way to evaluate constraints.
			"""
			constraint_eval_kwargs['theta'] = solution
			spec_orig = constraint_eval_kwargs['spec_orig']
			spec_for_experiment = constraint_eval_kwargs['spec_for_experiment']
			regime = constraint_eval_kwargs['regime']
			if regime == 'supervised_learning':
				# Use the original dataset as ground truth
				constraint_eval_kwargs['dataset']=spec_orig.dataset 

			if regime == 'reinforcement_learning':
				episodes_for_eval = constraint_eval_kwargs['episodes_for_eval']

				dataset_for_eval = RLDataSet(
					episodes=episodes_for_eval,
					meta_information=spec_for_experiment.dataset.meta_information,
					regime=regime)

				constraint_eval_kwargs['dataset'] = dataset_for_eval
	
			for parse_tree in spec_for_experiment.parse_trees:
				parse_tree.evaluate_constraint(
					**constraint_eval_kwargs)
				
				g = parse_tree.root.value
				if g > 0 or np.isnan(g):
					failed = True

		else:
			# User provided functions to evaluate constraints
			for eval_fn in constraint_eval_fns:
				g = eval_fn(solution)
				if g > 0 or np.isnan(g):
					failed = True
		return failed