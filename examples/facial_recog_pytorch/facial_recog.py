# generate_gpa_plots.py
import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss,accuracy_score

import torch

if __name__ == "__main__":
	# Parameter setup
	run_experiments = False
	make_plots = True
	save_plot = True
	include_legend = True
	performance_metric = 'Accuracy'
	# n_trials = 10
	n_trials = 10
	# data_fracs = np.logspace(-4,0,15)
	# data_fracs = [0.001,0.005,0.01,0.1,0.33,0.66,1.0] 
	data_fracs = [0.001,0.005,0.01,0.1,0.33,0.66,1.0]
	# batch_epoch_dict = {
	# 	0.001:[12,1000],
	# 	0.005:[60,1000],
	# 	0.01:[118,1000],
	# 	0.1:[237,200],
	# 	0.33:[230,50],
	# 	0.66:[237,30],
	# 	1.0: [237,20]
	# }
	batch_epoch_dict = {
		0.001:[24,50],
		0.005:[119,50],
		0.01:[237,75],
		0.1:[237,30],
		0.33:[237,20],
		0.66:[237,10],
		1.0: [237,10]
	}
	
	n_workers = 1
	results_dir = f'../../results/facial_recog_2022Dec19'
	plot_savename = os.path.join(results_dir,f'facial_recog_{performance_metric}.png')

	verbose=True

	# Load spec
	specfile = f'../../../engine-repo-dev/examples/facial_recog_pytorch/spec.pkl'
	spec = load_pickle(specfile)

	os.makedirs(results_dir,exist_ok=True)

	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	test_features = dataset.features
	test_labels = dataset.labels

	# Setup performance evaluation function and kwargs 
	# of the performance evaluation function

	def perf_eval_fn(y_pred,y,**kwargs):
		print("y_pred:")
		print(y_pred)
		if performance_metric == 'log_loss':
			return log_loss(y,y_pred)
		elif performance_metric == 'Accuracy':
			return accuracy_score(y,y_pred > 0.5)
	device = spec.model.device
	
	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		'device':device,
		'eval_batch_size':2000
		}

	constraint_eval_kwargs = {
		'eval_batch_size':2000
		}

	plot_generator = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		n_workers=n_workers,
		datagen_method='resample',
		perf_eval_fn=perf_eval_fn,
		constraint_eval_fns=[],
		constraint_eval_kwargs=constraint_eval_kwargs,
		results_dir=results_dir,
		perf_eval_kwargs=perf_eval_kwargs,
		batch_epoch_dict=batch_epoch_dict,
		)

	# # Baseline models
	if run_experiments:
		# plot_generator.run_baseline_experiment(
		# 	model_name='random_classifier',verbose=True)

		plot_generator.run_baseline_experiment(
			model_name='facial_recog_cnn',verbose=True)

		# Seldonian experiment
		# plot_generator.run_seldonian_experiment(verbose=verbose)


	if make_plots:
		if save_plot:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric,
				include_legend=include_legend,
				savename=plot_savename)
		else:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				include_legend=include_legend,
				performance_label=performance_metric)