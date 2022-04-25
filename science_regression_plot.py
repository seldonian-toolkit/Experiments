import os
import glob
import time
from functools import partial

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.dataset import *
from src.model import *
from src.candidate_selection import CandidateSelection
from src.safety_test import SafetyTest
from src.parse_tree import ParseTree
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

sensitive_column_names = ['M','F']
label_column = 'GPA'
plot_dir = './regression_plots'

def linear_regression(features,labels,sample_id):
	# fit to training data, predict on left out datapoint
	# Calculate prediction error on each prediction
	# Then take the mean and standard deviation 
	# of the prediction errors for male vs. female 
	train_features = features.drop(sample_id)
	train_labels = labels.drop(sample_id)
	test_features = features.iloc[sample_id]
	label = labels.iloc[sample_id]
	lr_model = LinearRegressionModel()
	theta = lr_model.fit(lr_model,train_features,train_labels)
	
	# prediction error = predicted GPA - true GPA
	predicted_gpa = lr_model.predict(theta,test_features)
	error = predicted_gpa - label
	
	return sample_id,predicted_gpa,label,error

def QNDLR(dataset,sample_id):
	# Run Seldonian on training data, predict using candidate solution 
	# on left out datapoint. If it passes the safety test, then
	# calculate prediction error on each prediction
	# Then take the mean and standard deviation 
	# of the prediction errors for male vs. female 
	
	# train_features = features.drop(sample_id)
	# train_labels = labels.drop(sample_id)
	features = dataset.df.loc[:,
		dataset.df.columns != label_column]
	features = features.drop(
		columns=dataset.sensitive_column_names)
	features.insert(0,'offset',1.0) # inserts a column of 1's
	test_features = features.iloc[sample_id]

	labels = dataset.df[label_column]
	test_label = labels.iloc[sample_id]

	dataset.df = dataset.df.drop(sample_id)
	
	candidate_df, safety_df = train_test_split(
			dataset.df, test_size=0.8, shuffle=False)

	candidate_dataset = DataSet(
		candidate_df,meta_information=dataset.df.columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column='GPA')

	safety_dataset = DataSet(
		safety_df,meta_information=dataset.df.columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column='GPA')
	
	n_safety = len(safety_df)

	# Linear regression model
	model_instance = LinearRegressionModel()

	# # Constraints
	constraint_str1 = 'abs(MED_MF) - 0.05'
	delta1 = 0.05

	# Create parse tree object
	pt1 = ParseTree(delta=delta1)

	# Fill out tree
	pt1.create_from_ast(s=constraint_str1)

	# assign delta to single node
	pt1.assign_deltas(weight_method='equal')

	# assign needed bounds
	pt1.assign_bounds_needed()

	parse_trees = [pt1]
	# print("made it here")

	minimizer_options = {}

	cs = CandidateSelection(
	    model=model_instance,
	    candidate_dataset=candidate_dataset,
	    n_safety=n_safety,
	    parse_trees=parse_trees,
	    primary_objective=model_instance.sample_Mean_Squared_Error,
	    optimization_technique='barrier_function',
	    optimizer='Nelder-Mead',
	    initial_solution_fn=model_instance.fit)

	candidate_solution = cs.run(minimizer_options=minimizer_options)

	st = SafetyTest(safety_dataset,model_instance,parse_trees)
	passed = st.run(candidate_solution,bound_method='ttest')

	# prediction error = predicted GPA - true GPA
	predicted_gpa = model_instance.predict(candidate_solution,test_features)
	error = predicted_gpa - test_label
	# Write out a file
	savename = os.path.join(plot_dir,
		f'results_qndlr_sample{str(sample_id).zfill(5)}.csv')
	result_df = pd.DataFrame([[sample_id,passed,predicted_gpa,test_label,error]])
	result_df.columns = ['sample_id','passed','predicted_gpa','label','error']
	result_df.to_csv(savename,index=False)
	print(f"Saved {savename}")
	return sample_id,passed,predicted_gpa,test_label,error

def run_experiment(sample_id):
	# Prepares file where experiment results will be saved
	experiment_number = worker_id
	outputFile = bin_path + 'results%d.npz' % experiment_number
	print("Writing output to", outputFile)
	
	# Generate the training data, D
	base_seed         = (experiment_number * numTrials)+1
	np.random.seed(base_seed+trial) # done to obtain common random numbers for all values of m			
	(trainX, trainY)  = generateData(m)

	# Run the Quasi-Seldonian algorithm
	(result, passedSafetyTest) = QSA(trainX, trainY, gHats, deltas)
	if passedSafetyTest:
		seldonian_solutions_found[trial, mIndex] = 1
		trueMSE = -fHat(result, testX, testY)                               # Get the "true" mean squared error using the testData
		seldonian_failures_g1[trial, mIndex] = 1 if trueMSE > 2.0  else 0   # Check if the first behavioral constraint was violated
		seldonian_failures_g2[trial, mIndex] = 1 if trueMSE < 1.25 else 0	# Check if the second behavioral constraint was violated
		seldonian_fs[trial, mIndex] = -trueMSE                              # Store the "true" negative mean-squared error
		print(f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial+1}/{numTrials}, m {m}] A solution was found: [{result[0]:.10f}, {result[1]:.10f}]\tfHat over test data: {trueMSE:.10f}")
	else:
		seldonian_solutions_found[trial, mIndex] = 0             # A solution was not found
		seldonian_failures_g1[trial, mIndex]     = 0             # Returning NSF means the first constraint was not violated
		seldonian_failures_g2[trial, mIndex]     = 0             # Returning NSF means the second constraint was not violated
		seldonian_fs[trial, mIndex]              = None          # This value should not be used later. We use None and later remove the None values
		print(f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial+1}/{numTrials}, m {m}] No solution found")

	# Run the Least Squares algorithm
	theta = leastSq(trainX, trainY)                              # Run least squares linear regression
	trueMSE = -fHat(theta, testX, testY)                         # Get the "true" mean squared error using the testData
	LS_failures_g1[trial, mIndex] = 1 if trueMSE > 2.0  else 0   # Check if the first behavioral constraint was violated
	LS_failures_g2[trial, mIndex] = 1 if trueMSE < 1.25 else 0   # Check if the second behavioral constraint was violated
	LS_fs[trial, mIndex] = -trueMSE                              # Store the "true" negative mean-squared error
	print(f"[(worker {worker_id}/{nWorkers}) LeastSq   trial {trial+1}/{numTrials}, m {m}] LS fHat over test data: {trueMSE:.10f}")

	np.savez(outputFile, 
			 ms=ms, 
			 seldonian_solutions_found=seldonian_solutions_found,
			 seldonian_fs=seldonian_fs, 
			 seldonian_failures_g1=seldonian_failures_g1, 
			 seldonian_failures_g2=seldonian_failures_g2,
			 LS_solutions_found=LS_solutions_found,
			 LS_fs=LS_fs,
			 LS_failures_g1=LS_failures_g1,
			 LS_failures_g2=LS_failures_g2)

if __name__ == "__main__":
	start = time.time()
	
	qndlr_files = sorted(
		glob.glob(plot_dir + '/results_qndlr_sample*csv'))

	csv_file = '../datasets/GPA/data_phil_modified.csv'
	columns = ["M","F","SAT_Physics",
		   "SAT_Biology","SAT_History",
		   "SAT_Second_Language","SAT_Geography",
		   "SAT_Literature","SAT_Portuguese_and_Essay",
		   "SAT_Math","SAT_Chemistry","GPA"]
	
	loader = DataSetLoader(column_names=columns,
		sensitive_column_names=sensitive_column_names,
		regime='supervised',label_column='GPA')

	dataset = loader.from_csv(csv_file)
	labels = dataset.df[label_column]
	sensitive_features = dataset.df.loc[:,
		dataset.sensitive_column_names]

	features = dataset.df.loc[:,
		dataset.df.columns != label_column]

	features = features.drop(
		columns=dataset.sensitive_column_names)

	features.insert(0,'offset',1.0) # inserts a column of 1's
	M_mask = sensitive_features['M'] == 1
	M_sample_ids = features.index[M_mask]

	errors_male = []
	errors_female = []
	for f in qndlr_files:
		df = pd.read_csv(f)
		sample_id = df['sample_id'].iloc[0]
		# print(sample_id)
		passed = df['passed'].iloc[0]
		error = df['error'].iloc[0]
		if not passed:
			continue

		if sample_id in M_sample_ids:
			# print("male sample id")
			errors_male.append(error)
		else:
			errors_female.append(error)
	print(len(errors_male))
	print(len(errors_female))
	mean_error_male = np.mean(errors_male)
	std_error_male = np.std(errors_male)/np.sqrt(len(errors_male))
	mean_error_female = np.mean(errors_female)
	std_error_female = np.std(errors_female)/np.sqrt(len(errors_female))
	print(mean_error_male,std_error_male)
	print(mean_error_female,std_error_female)
		# print([x for x in result])
		# result_df = pd.DataFrame([x for x in result])
		# result_df.columns = ['sample_id','passed','predicted_gpa','label','error']
	# 	result_df.to_csv(savename,index=False)
	# 	print(f"Saved {savename}")
	# 	# print(result_df)
	# 	end = time.time()
	# 	print(f"--- {end-start} seconds ---")

	# # Make the plot
	# result_df = pd.read_csv(savename)
	# # make male and female dataframes
	
	# result_df_M = result_df[M_mask]
	# result_df_F = result_df[~M_mask]
	# errors_M = result_df_M['error']
	# errors_F = result_df_F['error']

	# mean_error_M = np.mean(errors_M)
	# std_error_M = np.std(errors_M)/np.sqrt(len(features)-1)
	# print(mean_error_M,std_error_M)
	# mean_error_F = np.mean(errors_F)
	# std_error_F = np.std(errors_F)/np.sqrt(len(features)-1)
	# print(mean_error_F,std_error_F)
	

