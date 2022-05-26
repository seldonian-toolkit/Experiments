import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.utils import generate_resampled_datasets
from experiments.generate_plots import SupervisedPlotGenerator

from seldonian.utils.io_utils import load_pickle

def accuracy(model,theta,X,y):
	""" Binary classification accuracy """
	prediction = model.predict(theta,X)
	predict_class = prediction>=0.5
	acc = np.mean(1.0*predict_class==y)
	return acc


if __name__ == "__main__":
	# Load spec
	interface_output_dir = os.path.join('/Users/ahoag/beri/code',
		'interface_outputs/demographic_parity_interface_withspec')
	specfile = os.path.join(interface_output_dir,'spec.pkl')
	spec = load_pickle(specfile)

	### PARAMETER SETUP ###
	n_trials = 10
	data_pcts = np.hstack([[0.0004,0.0008],np.logspace(-3,0,15)])
	# data_pcts = [0.001]
	n_jobs=8
	results_dir = './results/demographic_parity_fromspec'
	verbose=True

	os.makedirs(results_dir,exist_ok=True)

	plot_generator = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_pcts=data_pcts,
		n_jobs=n_jobs,
		eval_method='resample',
		perf_eval_fn=accuracy,
		results_dir=results_dir,
		)

	# plot_generator.run_seldonian_experiment(verbose=verbose)

	savename = os.path.join(results_dir,'demographic_parity_withspec.png')
	plot_generator.make_plots(fontsize=12,legend_fontsize=8,
		performance_label='accuracy',best_performance=None,
		savename=savename)



