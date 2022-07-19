import os
import numpy as np
import pandas as pd

import pytest

from experiments.generate_plots import (
	SupervisedPlotGenerator)

def MSE(theta,model,X,y):
	n = len(X)
	prediction = model.predict(theta,X) # vector of values
	res = sum(pow(prediction-y,2))/n
	return res

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_supervised_plot_generator(gpa_regression_spec,experiment):
	np.random.seed(42)
	constraint_strs = ['Mean_Squared_Error - 3.0','2.0 - Mean_Squared_Error']
	deltas = [0.05,0.1]
	spec = gpa_regression_spec(constraint_strs,deltas)
	n_trials = 5
	# data_pcts = [0.001,0.05,0.5,1.0]
	data_pcts = [0.01,0.25]
	datagen_method="resample"
	perf_eval_fn = MSE
	results_dir = "./tests/static/results"
	n_workers = 1
	# Get performance evaluation kwargs set up
	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	label_column = dataset.label_column
	include_sensitive_columns = dataset.include_sensitive_columns
	include_intercept_term = dataset.include_intercept_term

	test_features = dataset.df.loc[:,
		dataset.df.columns != label_column]
	test_labels = dataset.df[label_column]

	if not include_sensitive_columns:
		test_features = test_features.drop(
			columns=dataset.sensitive_column_names)	

	if include_intercept_term:
		test_features.insert(0,'offset',1.0) # inserts a column of 1's in place

	# Define any additional keyword arguments (besides theta)
	# of the performance evaluation function,
	# which in our case is accuracy
	perf_eval_kwargs = {
		'model':spec.model_class(),
		'X':test_features,
		'y':test_labels,
		}
	
	spg = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_pcts=data_pcts,
		datagen_method=datagen_method,
		perf_eval_fn=perf_eval_fn,
		results_dir=results_dir,
		n_workers=n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs=perf_eval_kwargs,
		constraint_eval_kwargs={})
	
	assert spg.n_trials == n_trials
	assert spg.regime == 'supervised'

	spg.run_seldonian_experiment(verbose=True)

	## Make sure results file was created
	results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
	assert os.path.exists(results_file)

	# Make sure length of df is correct
	df = pd.read_csv(results_file)
	print(df)
	assert len(df) == 10
	dps = df.data_pct
	trial_is = df.trial_i
	perfs = df.performance
	passed_safetys = df.passed_safety
	faileds = df.failed
	
	assert dps[0] == 0.01
	assert trial_is[0] == 0
	assert np.isnan(perfs[0])
	assert passed_safetys[0] == False
	assert faileds[0] == False

	assert dps[4] == 0.01
	assert trial_is[4] == 4
	assert perfs[4] == pytest.approx(2.443076)
	assert passed_safetys[4] == True
	assert faileds[4] == False

	assert dps[5] == 0.25
	assert trial_is[5] == 0
	assert perfs[5] == pytest.approx(2.106480)
	assert passed_safetys[5] == True
	assert faileds[5] == False

	assert dps[9] == 0.25
	assert trial_is[9] == 4
	assert np.isnan(perfs[9])
	assert passed_safetys[9] == False
	assert faileds[9] == False
	
	# Make sure number of trial files created is correct
	trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
	trial_files = os.listdir(trial_dir)
	assert len(trial_files) == 10

	# Make sure the trial files have the right format
	trial_file_0 = os.path.join(trial_dir,trial_files[0])
	df_trial0 = pd.read_csv(trial_file_0)
	assert len(df_trial0) == 1

	# Now make plot
	savename = os.path.join(results_dir,"test_gpa_regression_plot.png")
	spg.make_plots(fontsize=12,legend_fontsize=8,
		performance_label='MSE',best_performance=None,
		savename=savename)
	# Make sure it was saved
	assert os.path.exists(savename)