# generate_gpa_plots.py
import os, math
import numpy as np 
import torch

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle

if __name__ == "__main__":
	# Parameter setup
	run_experiments = False
	make_plots = True
	save_plot = False
	include_legend = True
	performance_metric = 'Accuracy'
	model_label_dict = {
		'qsa':'QSA',
		'facial_recog_cnn_lr0p001': 'CNN baseline',
		'weighted_random_classifier': 'Weighted-random classifier'}

	n_trials = 40
	
	data_fracs = np.logspace(-3,0,15)
	niter = 1200 # how many iterations we want in each run. Overfitting happens with more than this.
	batch_size=237
	data_sizes=data_fracs*11850 # number of points used in candidate selectionin each data frac
	n_batches=data_sizes/batch_size # number of batches in each data frac
	n_batches=np.array([math.ceil(x) for x in n_batches])
	n_epochs_arr=niter/n_batches # number of epochs needed to get to 1200 iterations in each data frac
	n_epochs_arr = np.array([math.ceil(x) for x in n_epochs_arr])

	batch_epoch_dict = {data_fracs[ii]:[batch_size,n_epochs_arr[ii]] for ii in range(len(data_fracs))}
	n_workers = 1

	results_dir = f'../../results/faact_conference_fullmodel'
	# plot_savename = os.path.join(results_dir,f'fullCNN_experiment.pdf')
	plot_savename = os.path.join(results_dir,f'facial_gender_experiment4tutorials.png')
	verbose=False
	# Load spec
	specfile = './full_cnn_spec.pkl'
	spec = load_pickle(specfile)
	# Define batch epoch dict for training baseline CNN

	os.makedirs(results_dir,exist_ok=True)

	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	test_features = dataset.features
	test_labels = dataset.labels

	# Setup performance evaluation function and kwargs 
	# of the performance evaluation function

	def perf_eval_fn(y_pred,y,**kwargs):
		if performance_metric == 'Accuracy':
			# 1 - error rate
			v = np.where(y!=1.0,1.0-y_pred,y_pred)
			return sum(v)/len(v)

	# device = torch.device("cpu")
	# spec.model.device = device
	# spec.model.pytorch_model.to(device)

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
		batch_epoch_dict=batch_epoch_dict
		)

	# # Baseline models
	if run_experiments:
		# Seldonian experiment
		plot_generator.run_seldonian_experiment(verbose=verbose)


	if make_plots:	
		plot_generator.make_plots(
			model_label_dict=model_label_dict,fontsize=12,legend_fontsize=8,
			performance_label=performance_metric,
			performance_ylims=[0,1],
			save_format='png',
			show_title=True,
			custom_title=r'Constraint: $\operatorname{min}\left(\frac{ACC|[M]}{ACC|[F]},\frac{ACC|[F]}{ACC|[M]}\right) \geq 0.8$',
			include_legend=include_legend,
			savename=plot_savename if save_plot else None)
	