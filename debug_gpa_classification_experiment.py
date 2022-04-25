import os
import time
from functools import partial

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from seldonian.dataset import *
from seldonian.model import *
from seldonian.candidate_selection import CandidateSelection
from seldonian.safety_test import SafetyTest
from seldonian.parse_tree import ParseTree
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

n_trials=250
sensitive_column_names = ['M','F']
label_column = 'GPA_class'
plot_dir = './classification_plots'
frac_data_in_safety = 0.6
delta = 0.05 # for all seldonian classification experiments

def generate_resampled_datasets(df,with_negative_labels=False):
	for trial_i in range(250):
		if with_negative_labels:
			savename = os.path.join(plot_dir,
				'resampled_dataframes_with_negative_labels',
				f'trial_{trial_i}.csv')
		else:
			savename = os.path.join(plot_dir,
				'resampled_dataframes',
				f'trial_{trial_i}.csv')

		if not os.path.exists(savename):
			resampled_df = df.sample(n=len(df),replace=True)
			if with_negative_labels:
				if 0.0 in resampled_df[label_column]:
					resampled_df.loc[resampled_df[label_column]==0.0,label_column]=-1.0
			else:
				if -1.0 in resampled_df[label_column]:
					resampled_df.loc[resampled_df[label_column]==-1.0,label_column]=0.0
			resampled_df.to_csv(savename,index=False)

def run_sgd(data_pcts,n_trials,
	M_mask,orig_features,orig_labels):
	# set -1 to 0 in labels, makes logistic regression more convenient
	orig_labels = orig_labels.copy()
	orig_labels.loc[orig_labels==-1]=0

	result_list = []
	helper = partial(sgd,
		M_mask=M_mask,
		orig_features=orig_features,
		orig_labels=orig_labels)
	data_pcts_vector = np.array([x for x in data_pcts for y in range(n_trials)])
	trials_vector = np.array([x for y in range(len(data_pcts)) for x in range(n_trials)])
	# helper(0.05,0)
	with ProcessPoolExecutor(max_workers=8) as ex:
		for ii,res in enumerate(ex.map(helper,data_pcts_vector,trials_vector)):
			print(res)
			# acc,failed_di,failed_dp,failed_eo,failed_eodds,failed_pe = res
			# result_list.append([
			# 	data_pcts_vector[ii],
			# 	trials_vector[ii],
			# 	acc,
			# 	failed_di,
			# 	failed_dp,
			# 	failed_eo,
			# 	failed_eodds,
			# 	failed_pe])

	# result_df = pd.DataFrame(result_list)
	# # print(result_df)
	# result_df.columns = ['data_pct',
	# 	'trial_i','accuracy',
	# 	'failed_di','failed_dp','failed_eo',
	# 	'failed_eodds','failed_pe']
	return 

def sgd(
	data_pct,
	trial_i,
	M_mask,
	orig_features,
	orig_labels):

	savename = os.path.join(plot_dir,
		f'sgd_results',
		f'data_pct_{data_pct}_trial_{trial_i}.csv')
	if os.path.exists(savename):
		return 

	df_filename = os.path.join(plot_dir,'resampled_dataframes',f'trial_{trial_i}.csv')
	n_this_pct = int(round(len(orig_features)*data_pct))

	resampled_df = pd.read_csv(df_filename)[:n_this_pct]

	features = resampled_df.loc[:,
		resampled_df.columns != label_column]
	features = features.drop(
		columns=sensitive_column_names)

	labels = resampled_df[label_column]
	
	clf = make_pipeline(StandardScaler(),
              SGDClassifier(max_iter=1000, tol=1e-3))

	clf.fit(features, labels)

	# Calculate accuracy from entire original dataset 
	prediction = clf.predict(orig_features)

	acc = np.mean(1.0*prediction==orig_labels)
	print(data_pct,trial_i)
	# Determine whether this solution
	# violates the constraints on the original dataset
	
	prediction_M = prediction[M_mask]
	prediction_F = prediction[~M_mask]
	PR_M = np.mean(prediction_M==1)
	PR_F = np.mean(prediction_F==1)
	
	orig_labels_M = orig_labels[M_mask]
	orig_labels_F = orig_labels[~M_mask]
	FPR_M = np.mean(
		np.logical_and(
			orig_labels_M!=1.0,
			prediction_M==1.0)
		)
	FPR_F = np.mean(
		np.logical_and(
			orig_labels_F!=1.0,
			prediction_F==1.0)
		)

	FNR_M = np.mean(
		np.logical_and(
			orig_labels_M==1.0,
			prediction_M!=1.0)
		)
	FNR_F = np.mean(
		np.logical_and(
			orig_labels_F==1.0,
			prediction_F!=1.0)
		)
	TPR_M = np.mean(
		np.logical_and(
			orig_labels_M==1.0,
			prediction_M==1.0)
		)
	TPR_F = np.mean(
		np.logical_and(
			orig_labels_F==1.0,
			prediction_F==1.0)
		)

	# Disparate impact 
	failed_di = False
	ghat_di = 0.8 - min(PR_M/PR_F,PR_F/PR_M)
	if ghat_di > 0:
		failed_di = True

	# Demographic parity
	failed_dp = False
	ghat_dp = abs(PR_M-PR_F) - 0.15
	if ghat_dp > 0:
		failed_dp = True

	# Equal opportunity	
	
	failed_eo = False
	ghat_eo = abs(FNR_M-FNR_F) - 0.2
	if ghat_eo > 0:
		failed_eo = True	
	
	# Equalized odds
	failed_eodds = False
	ghat_eodds = abs(FNR_M-FNR_F) + abs(FPR_M-FPR_F) - 0.35
	if ghat_eodds > 0:
		failed_eodds = True

	# Equalized odds - Stephen's version
	failed_eodds_stephen = False
	ghat_eodds_stephen = abs(TPR_M-TPR_F) + abs(FPR_M-FPR_F) - 0.35
	if ghat_eodds_stephen > 0:
		failed_eodds_stephen = True

	# Predictive equality
	failed_pe = False
	ghat_pe = abs(FPR_M-FPR_F) - 0.2

	if ghat_pe > 0:
		failed_pe = True

	# Write out file for this data_pct,trial_i combo
	result_df = pd.DataFrame([[data_pct,trial_i,acc,
		TPR_M,TPR_F,FPR_M,FPR_F,FNR_M,FNR_F,PR_M,PR_F,failed_di,
		failed_dp,failed_eo,failed_eodds,
		failed_eodds_stephen,failed_pe]])
	result_df.columns = ['data_pct','trial_i','accuracy',
	'TPR_M','TPR_F','FPR_M','FPR_F','FNR_M','FNR_F',
	'PR_M','PR_F','failed_di','failed_dp','failed_eo',
	'failed_eodds','failed_eodds_stephen','failed_pe']
	result_df.to_csv(savename,index=False)
	print(f"Saved {savename}")
	return "Success"

def run_logistic_regression(data_pcts,n_trials,
	M_mask,orig_features,orig_labels):
	# set -1 to 0 in labels, makes logistic regression more convenient
	orig_labels = orig_labels.copy()
	orig_labels.loc[orig_labels==-1]=0

	result_list = []
	helper = partial(logistic_regression,
		M_mask=M_mask,
		orig_features=orig_features,
		orig_labels=orig_labels)

	data_pcts_vector = np.array([x for x in data_pcts for y in range(n_trials)])
	trials_vector = np.array([x for y in range(len(data_pcts)) for x in range(n_trials)])

	with ProcessPoolExecutor(max_workers=8) as ex:
		for ii,res in enumerate(ex.map(helper,data_pcts_vector,trials_vector)):
			print(res)
			# acc,failed_di,failed_dp,failed_eo,failed_eodds,failed_pe = res
			# result_list.append([
			# 	data_pcts_vector[ii],
			# 	trials_vector[ii],
			# 	acc,
			# 	failed_di,
			# 	failed_dp,
			# 	failed_eo,
			# 	failed_eodds,
			# 	failed_pe])

	# result_df = pd.DataFrame(result_list)
	# # print(result_df)
	# result_df.columns = ['data_pct',
	# 	'trial_i','accuracy',
	# 	'failed_di','failed_dp','failed_eo',
	# 	'failed_eodds','failed_pe']
	return 

def logistic_regression(
	data_pct,
	trial_i,
	M_mask,
	orig_features,
	orig_labels):
	savename = os.path.join(plot_dir,
		f'logistic_regression_results',
		f'data_pct_{data_pct}_trial_{trial_i}.csv')
	if os.path.exists(savename):
		return 

	df_filename = os.path.join(plot_dir,'resampled_dataframes',f'trial_{trial_i}.csv')
	n_this_pct = int(round(len(orig_features)*data_pct))

	resampled_df = pd.read_csv(df_filename)[:n_this_pct]

	features = resampled_df.loc[:,
		resampled_df.columns != label_column]
	features = features.drop(
		columns=sensitive_column_names)

	labels = resampled_df[label_column]
	
	lr_model = LogisticRegression(max_iter=200)
	theta = lr_model.fit(features, labels)
	
	# Calculate accuracy from entire original dataset 
	prediction = lr_model.predict(orig_features)

	acc = np.mean(1.0*prediction==orig_labels)
	print(data_pct,trial_i)
	# Determine whether this solution
	# violates the constraints on the original dataset
	
	prediction_M = prediction[M_mask]
	prediction_F = prediction[~M_mask]
	PR_M = np.mean(prediction_M==1)
	PR_F = np.mean(prediction_F==1)
	
	orig_labels_M = orig_labels[M_mask]
	orig_labels_F = orig_labels[~M_mask]
	FPR_M = np.mean(
		np.logical_and(
			orig_labels_M!=1.0,
			prediction_M==1.0)
		)
	FPR_F = np.mean(
		np.logical_and(
			orig_labels_F!=1.0,
			prediction_F==1.0)
		)

	FNR_M = np.mean(
		np.logical_and(
			orig_labels_M==1.0,
			prediction_M!=1.0)
		)
	FNR_F = np.mean(
		np.logical_and(
			orig_labels_F==1.0,
			prediction_F!=1.0)
		)
	TPR_M = np.mean(
		np.logical_and(
			orig_labels_M==1.0,
			prediction_M==1.0)
		)
	TPR_F = np.mean(
		np.logical_and(
			orig_labels_F==1.0,
			prediction_F==1.0)
		)

	# Disparate impact 
	failed_di = False
	ghat_di = 0.8 - min(PR_M/PR_F,PR_F/PR_M)
	if ghat_di > 0:
		failed_di = True

	# Demographic parity
	failed_dp = False
	ghat_dp = abs(PR_M-PR_F) - 0.15
	if ghat_dp > 0:
		failed_dp = True

	# Equal opportunity	
	
	failed_eo = False
	ghat_eo = abs(FNR_M-FNR_F) - 0.2
	if ghat_eo > 0:
		failed_eo = True	
	
	# Equalized odds
	failed_eodds = False
	ghat_eodds = abs(FNR_M-FNR_F) + abs(FPR_M-FPR_F) - 0.35
	if ghat_eodds > 0:
		failed_eodds = True

	# Equalized odds - Stephen's version
	failed_eodds_stephen = False
	ghat_eodds_stephen = abs(TPR_M-TPR_F) + abs(FPR_M-FPR_F) - 0.35
	if ghat_eodds_stephen > 0:
		failed_eodds_stephen = True

	# Predictive equality
	failed_pe = False
	ghat_pe = abs(FPR_M-FPR_F) - 0.2

	if ghat_pe > 0:
		failed_pe = True

	# Write out file for this data_pct,trial_i combo
	result_df = pd.DataFrame([[data_pct,trial_i,acc,
		TPR_M,TPR_F,FPR_M,FPR_F,FNR_M,FNR_F,PR_M,PR_F,failed_di,
		failed_dp,failed_eo,failed_eodds,
		failed_eodds_stephen,failed_pe]])
	result_df.columns = ['data_pct','trial_i','accuracy',
	'TPR_M','TPR_F','FPR_M','FPR_F','FNR_M','FNR_F',
	'PR_M','PR_F','failed_di','failed_dp','failed_eo',
	'failed_eodds','failed_eodds_stephen','failed_pe']
	result_df.to_csv(savename,index=False)
	print(f"Saved {savename}")
	return "Success"

def run_seldonian_experiment(
	constraint,
	data_pcts,n_trials,
	orig_dataset,
	M_mask,
	orig_features,orig_labels):
	result_list = []

	# args that don't change are: constraint, orig dataset,
	# orig_features, orig_labels, resampled_datasets
	helper = partial(seldonian_classification,
		constraint=constraint,
		orig_dataset=orig_dataset,
		M_mask=M_mask,
		orig_features=orig_features,
		orig_labels=orig_labels)
	
	# Need to pass map a flattened list of data_pcts like:
	# [0.005,0.005,...250times,0.01,0.01,...250times,...,1.0,1.0,...250times]
	data_pcts_vector = np.array([x for x in data_pcts for y in range(n_trials)])
	trials_vector = np.array([x for y in range(len(data_pcts)) for x in range(n_trials)])
	for i in range(1):
		helper(1.0,i*5)
	# with ProcessPoolExecutor(max_workers=8) as ex:
	# 	result = ex.map(helper,data_pcts_vector,trials_vector)
	# 	for res in result:
	# 		print(res)
	
def seldonian_classification(
	data_pct,
	trial_i,
	constraint,
	orig_dataset,
	M_mask,
	orig_features,
	orig_labels):

	savename = os.path.join(plot_dir,
		f'{constraint}_results',
		f'data_pct_{data_pct}_trial_{trial_i}.csv')
	# if os.path.exists(savename):
	# 	return 

	# Load resampled dataframe for this trial
	df_filename = os.path.join(plot_dir,
		'resampled_dataframes_with_negative_labels',
		f'trial_{trial_i}.csv')
	
	n_this_pct = int(round(len(orig_features)*data_pct))

	resampled_df = pd.read_csv(df_filename)[:n_this_pct]

	if constraint == 'disparate_impact':
		constraint_str = '0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'
	elif constraint == 'demographic_parity':
		constraint_str = 'abs((PR | [M]) - (PR | [F])) - 0.15'
	elif constraint == 'equal_opportunity':
		constraint_str = 'abs((FNR | [M]) - (FNR | [F])) - 0.2'
	elif constraint == 'equalized_odds':
		constraint_str = 'abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) - 0.35'
	elif constraint == 'equalized_odds_stephen':
		constraint_str = 'abs((TPR | [M]) - (TPR | [F])) + abs((FPR | [M]) - (FPR | [F])) - 0.35'
	elif constraint == 'predictive_equality':
		constraint_str = 'abs((FPR | [M]) - (FPR | [F])) - 0.2'
	else:
		sys.exit("Bad constraint provided: ", constraint)

	# Create parse tree object
	pt = ParseTree(delta=delta)
	
	pt.create_from_ast(constraint_str)
	
	# assign deltas for each base node
	# use equal weighting for each base node
	pt.assign_deltas(weight_method='equal')

	# Assign bounds needed on the base nodes
	pt.assign_bounds_needed()

	parse_trees = [pt]

	candidate_df, safety_df = train_test_split(
			resampled_df, test_size=frac_data_in_safety, shuffle=False)
	
	sensitive_candidate_df = candidate_df[sensitive_column_names]
	sensitive_safety_df = safety_df[sensitive_column_names]
	
	candidate_dataset = DataSet(
		candidate_df,meta_information=resampled_df.columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)

	safety_dataset = DataSet(
		safety_df,meta_information=resampled_df.columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)
	
	n_safety = len(safety_df)

	# Linear regression model for classification
	model = LinearClassifierModel()
	# model = SGDClassifierModel()
	# model_instance = LogisticRegressionModel()

	# Candidate selection
	minimizer_options = {}
	
	cs = CandidateSelection(
		model=model,
		candidate_dataset=candidate_dataset,
		n_safety=n_safety,
		parse_trees=parse_trees,
		primary_objective=model.perceptron_loss,
		optimization_technique='barrier_function',
		# optimizer='Nelder-Mead',
		optimizer='Powell',
		# optimizer='CMA-ES',
		initial_solution_fn=model.fit)

	# candidate_solution = cs.run(minimizer_options=minimizer_options)
	
	# candidate_solution = np.array([-6.33906378e+00, 2.12614294e-04,  4.58789139e-04,  1.85429490e-03,  1.43927804e-03,
 #  5.23958424e-04,  3.31554578e-03,  3.01613714e-03, -2.71727612e-04,
 # -1.56083616e-04,])
	candidate_solution = np.array([-5.34202688e+00, 1.90213674e-05,  9.23978779e-04,  2.93852361e-03,  1.12053706e-03,
 -3.35866543e-03,  7.27481640e-04,  5.81141352e-03,  5.15192864e-04,
 -1.73628101e-03,])

	# Safety test
	st = SafetyTest(safety_dataset,model,parse_trees)
	passed_safety = st.run(candidate_solution,bound_method='ttest')
	# Calculate accuracy from entire original dataset 

	prediction = model.predict(
		candidate_solution,
		orig_features)
	acc = np.mean(1.0*prediction==orig_labels)
	# Calculate whether we failed, i.e. 
	# we said we passed the safety test but
	# the constraint fails on the original dataset (ground truth)
	failed=False
	if passed_safety:
		# constraint_str = 'abs((PR | [M]) - (PR | [F])) - 0.15'
		prediction_M = prediction[M_mask]
		prediction_F = prediction[~M_mask]

		# only calculate the statistics that we need for each constraint
		if constraint in ['disparate_impact','demographic_parity']:
			PR_M = np.mean(prediction_M==1.0)
			PR_F = np.mean(prediction_F==1.0)

		if constraint == 'equal_opportunity':
			orig_labels_M = orig_labels[M_mask]
			orig_labels_F = orig_labels[~M_mask]
			FNR_M = np.mean(
				np.logical_and(
					orig_labels_M==1.0,
					prediction_M!=1.0)
				)
			FNR_F = np.mean(
				np.logical_and(
					orig_labels_F==1.0,
					prediction_F!=1.0)
				)
		
		if constraint == 'equalized_odds':
			orig_labels_M = orig_labels[M_mask]
			orig_labels_F = orig_labels[~M_mask]
			FNR_M = np.mean(
				np.logical_and(
					orig_labels_M==1.0,
					prediction_M!=1.0)
				)
			FNR_F = np.mean(
				np.logical_and(
					orig_labels_F==1.0,
					prediction_F!=1.0)
				)
			FPR_M = np.mean(
				np.logical_and(
					orig_labels_M!=1.0,
					prediction_M==1.0)
				)
			FPR_F = np.mean(
				np.logical_and(
					orig_labels_F!=1.0,
					prediction_F==1.0)
				)
		if constraint == 'equalized_odds_stephen':
			orig_labels_M = orig_labels[M_mask]
			orig_labels_F = orig_labels[~M_mask]
			TPR_M = np.mean(
				np.logical_and(
					orig_labels_M==1.0,
					prediction_M==1.0)
				)
			TPR_F = np.mean(
				np.logical_and(
					orig_labels_F==1.0,
					prediction_F==1.0)
				)
			FPR_M = np.mean(
				np.logical_and(
					orig_labels_M!=1.0,
					prediction_M==1.0)
				)
			FPR_F = np.mean(
				np.logical_and(
					orig_labels_F!=1.0,
					prediction_F==1.0)
				)

		if constraint == 'predictive_equality':
			orig_labels_M = orig_labels[M_mask]
			orig_labels_F = orig_labels[~M_mask]

			FPR_M = np.mean(
				np.logical_and(
					orig_labels_M!=1.0,
					prediction_M==1.0)
				)
			FPR_F = np.mean(
				np.logical_and(
					orig_labels_F!=1.0,
					prediction_F==1.0)
				)

		if constraint == 'disparate_impact':
			ghat = -0.8-min(PR_M/PR_F,PR_F/PR_M)
		elif constraint == 'demographic_parity':
			ghat = abs(PR_M-PR_F) - 0.15
		elif constraint == 'equal_opportunity':
			ghat = abs(FNR_M-FNR_F) - 0.2
		elif constraint == 'equalized_odds':
			ghat = abs(FNR_M-FNR_F) + abs(FPR_M-FPR_F) - 0.35
		elif constraint == 'equalized_odds_stephen':
			ghat = abs(TPR_M-TPR_F) + abs(FPR_M-FPR_F) - 0.35
		elif constraint == 'predictive_equality':
			ghat = abs(FPR_M-FPR_F) - 0.2

		if ghat > 0:
			failed = True
	print(data_pct,trial_i,acc,passed_safety,failed)	
	# Write out file for this data_pct,trial_i combo
	# result_df = pd.DataFrame([[data_pct,trial_i,acc,passed_safety,failed]])
	# result_df.columns = ['data_pct','trial_i','accuracy','passed_safety','failed']
	# result_df.to_csv(savename,index=False)
	# print(f"Saved {savename}")
	return "Success"

def aggregate_results(constraint,data_pcts,n_trials,savename):
	df_list = []
	for data_pct in data_pcts:
		for trial_i in range(n_trials):
			filename = os.path.join(plot_dir,
				f'{constraint}_results',
				f'data_pct_{data_pct}_trial_{trial_i}.csv')
			df = pd.read_csv(filename)
			df_list.append(df)
	result_df = pd.concat(df_list)
	result_df.to_csv(savename,index=False)

if __name__ == "__main__":
	start = time.time()
	np.random.seed(42)

	csv_file = '../datasets/GPA/data_classification.csv'
	columns = ["M","F","SAT_Physics",
		   "SAT_Biology","SAT_History",
		   "SAT_Second_Language","SAT_Geography",
		   "SAT_Literature","SAT_Portuguese_and_Essay",
		   "SAT_Math","SAT_Chemistry","GPA_class"]

	sensitive_column_names = ['M','F']
	label_column = "GPA_class"
	loader = DataSetLoader(column_names=columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column=label_column)
	dataset = loader.from_csv(csv_file)
	orig_features = dataset.df.loc[:,
		dataset.df.columns != label_column]
	M_mask = orig_features['M'] == 1
	orig_features = orig_features.drop(
		columns=sensitive_column_names)

	orig_labels = dataset.df[label_column]

	# generate_resampled_datasets(dataset.df)
	generate_resampled_datasets(dataset.df,with_negative_labels=True)

	# Resample from datasets at values of m, 250 trials per m.
	# Values of m taken from Stephen Giguere's code: https://github.com/sgiguere/SeldonianML/blob/science_1019/Python/experiments/scripts/science_experiments_brazil.bat
	# But it looks like the Science plots go down to lower numbers so I added 0.005
	data_pcts = [0.005,0.01, 0.012742749857, 0.0162377673919,
	 0.0206913808111, 0.0263665089873, 0.0335981828628,
	 0.0428133239872, 0.0545559478117, 0.0695192796178,
	 0.088586679041, 0.112883789168, 0.143844988829,
	 0.183298071083, 0.233572146909, 0.297635144163,
	 0.379269019073, 0.483293023857, 0.615848211066,
	 0.784759970351, 1.0]
	
	# Logistic regression
	savename_logistic_regression = os.path.join(
		plot_dir,"results_logistic_regression.csv")

	if not os.path.exists(savename_logistic_regression):
		run_logistic_regression(
			M_mask=M_mask,
			data_pcts=data_pcts,
			n_trials=n_trials,
			orig_features=orig_features,orig_labels=orig_labels)
		aggregate_results(constraint='logistic_regression',
			data_pcts=data_pcts,n_trials=n_trials,
			savename=savename_logistic_regression)
	
	# SGD - Linear SMV with Hinge Loss
	savename_sgd = os.path.join(
		plot_dir,"results_sgd.csv")

	if not os.path.exists(savename_sgd):
		run_sgd(
			M_mask=M_mask,
			data_pcts=data_pcts,
			n_trials=n_trials,
			orig_features=orig_features,orig_labels=orig_labels)

		aggregate_results(constraint='sgd',
			data_pcts=data_pcts,n_trials=n_trials,
			savename=savename_sgd)

	# Disparate impact
	savename_disparate_impact = os.path.join(
		plot_dir,"results_disparate_impact.csv")

	if not os.path.exists(savename_disparate_impact):
		orig_features.insert(0,'offset',1.0) # inserts a column of 1's
		run_seldonian_experiment(
			constraint='disparate_impact',data_pcts=data_pcts,
			n_trials=n_trials,
			orig_dataset=dataset,
			M_mask=M_mask,
			orig_features=orig_features,orig_labels=orig_labels)

		aggregate_results(
			constraint='disparate_impact',
			data_pcts=data_pcts,n_trials=n_trials,
			savename=savename_disparate_impact)

	# Demographic parity
	savename_demographic_parity = os.path.join(
		plot_dir,"results_demographic_parity.csv")

	if not os.path.exists(savename_demographic_parity):
		orig_features.insert(0,'offset',1.0) # inserts a column of 1's
		run_seldonian_experiment(
			constraint='demographic_parity',data_pcts=data_pcts,
			n_trials=n_trials,
			orig_dataset=dataset,
			M_mask=M_mask,
			orig_features=orig_features,orig_labels=orig_labels)

		aggregate_results(constraint='demographic_parity',
			data_pcts=data_pcts,n_trials=n_trials,
			savename=savename_demographic_parity)

	# Equal opportunity
	savename_equal_opportunity = os.path.join(
		plot_dir,"results_equal_opportunity.csv")

	# if not os.path.exists(savename_equal_opportunity):
	orig_features.insert(0,'offset',1.0) # inserts a column of 1's
	run_seldonian_experiment(
		constraint='equal_opportunity',data_pcts=data_pcts,
		n_trials=n_trials,
		orig_dataset=dataset,
		M_mask=M_mask,
		orig_features=orig_features,orig_labels=orig_labels)

		# aggregate_results(
		# 	constraint='equal_opportunity',
		# 	data_pcts=data_pcts,n_trials=n_trials,
		# 	savename=savename_equal_opportunity)

	# Equalized odds
	savename_equalized_odds = os.path.join(
		plot_dir,"results_equalized_odds.csv")

	if not os.path.exists(savename_equalized_odds):
		orig_features.insert(0,'offset',1.0) # inserts a column of 1's
		run_seldonian_experiment(
			constraint='equalized_odds',data_pcts=data_pcts,
			n_trials=n_trials,
			orig_dataset=dataset,
			M_mask=M_mask,
			orig_features=orig_features,orig_labels=orig_labels)

		aggregate_results(
			constraint='equalized_odds',
			data_pcts=data_pcts,n_trials=n_trials,
			savename=savename_equalized_odds)

	# Equalized odds (Stephen's version)
	savename_equalized_odds_stephen = os.path.join(
		plot_dir,"results_equalized_odds_stephen.csv")

	if not os.path.exists(savename_equalized_odds_stephen):
		orig_features.insert(0,'offset',1.0) # inserts a column of 1's
		run_seldonian_experiment(
			constraint='equalized_odds_stephen',data_pcts=data_pcts,
			n_trials=n_trials,
			orig_dataset=dataset,
			M_mask=M_mask,
			orig_features=orig_features,orig_labels=orig_labels)

		aggregate_results(
			constraint='equalized_odds_stephen',
			data_pcts=data_pcts,n_trials=n_trials,
			savename=savename_equalized_odds_stephen)

	# Predictive equality
	savename_predictive_equality = os.path.join(
		plot_dir,"results_predictive_equality.csv")

	if not os.path.exists(savename_predictive_equality):
		orig_features.insert(0,'offset',1.0) # inserts a column of 1's
		run_seldonian_experiment(
			constraint='predictive_equality',data_pcts=data_pcts,
			n_trials=n_trials,
			orig_dataset=dataset,
			M_mask=M_mask,
			orig_features=orig_features,orig_labels=orig_labels)

		aggregate_results(
			constraint='predictive_equality',
			data_pcts=data_pcts,n_trials=n_trials,
			savename=savename_predictive_equality)

	# end = time.time()
	# print(f"--- {end-start} seconds ---")	
	

	

