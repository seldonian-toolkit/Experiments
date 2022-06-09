import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.utils import generate_resampled_datasets
from experiments.generate_plots import SupervisedPlotGenerator

from seldonian.utils.io_utils import load_pickle

def accuracy(theta,model,X,y):
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
	spec.primary_objective = spec.model_class().sample_logistic_loss

	### PARAMETER SETUP ###
	n_trials = 10
	data_pcts = np.logspace(-4,0,15)
	# data_pcts = [0.01]
	n_workers=8
	results_dir = './results/demographic_parity_fromspec_test'
	verbose=True

	os.makedirs(results_dir,exist_ok=True)

	plot_generator = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_pcts=data_pcts,
		n_workers=n_workers,
		datagen_method='resample',
		perf_eval_fn=accuracy,
		constraint_eval_fns=[],
		results_dir=results_dir,
		)

	plot_generator.run_seldonian_experiment(verbose=verbose)

	savename = os.path.join(results_dir,'demographic_parity_withspec.png')
	plot_generator.make_plots(fontsize=12,legend_fontsize=8,
		performance_label='accuracy',best_performance=None,
		savename=savename)



