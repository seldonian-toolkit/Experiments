import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.utils import generate_resampled_datasets
from experiments.generate_plots import RLPlotGenerator

from seldonian.utils.io_utils import load_pickle

# def accuracy(model,theta,X,y):
# 	""" Binary classification accuracy """
# 	prediction = model.predict(theta,X)
# 	predict_class = prediction>=0.5
# 	acc = np.mean(1.0*predict_class==y)
# 	return acc

if __name__ == "__main__":
	# Load spec
	interface_output_dir = os.path.join('/Users/ahoag/beri/code',
		'interface_outputs/gridworld_fromspec')
	specfile = os.path.join(interface_output_dir,'spec.pkl')
	spec = load_pickle(specfile)
	print(dir(spec))
	RL_environment_obj = spec.RL_environment_obj
	perf_eval_fn = RL_environment_obj.calc_J

	### PARAMETER SETUP ###
	n_trials = 10
	# data_pcts = np.hstack([[0.0004,0.0008],np.logspace(-3,0,15)])
	data_pcts = np.logspace(-3,0,10)
	# data_pcts = [0.00025,0.0005]
	# data_pcts = [0.005,]
	n_workers=8
	n_episodes_for_eval=1000
	results_dir = './results/gridworld_fromspec_v2'
	verbose=True

	os.makedirs(results_dir,exist_ok=True)
	
	spec.optimization_hyperparams['num_iters'] = 20
	spec.optimization_hyperparams['alpha_theta'] = 0.05
	spec.optimization_hyperparams['alpha_lamb'] = 0.05
	# spec.regularization_hyperparams['reg_coef'] = 0.1

	plot_generator = RLPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_pcts=data_pcts,
		n_workers=n_workers,
		datagen_method='generate_episodes',
		n_episodes_for_eval=n_episodes_for_eval,
		RL_environment_obj=RL_environment_obj,
		perf_eval_fn=perf_eval_fn,
		constraint_eval_fns=[],
		results_dir=results_dir,
		)
	
	plot_generator.run_seldonian_experiment(verbose=verbose)

	# savename = os.path.join(results_dir,'gridworld_withspec.png')
	# plot_generator.make_plots(fontsize=12,legend_fontsize=8,
	# 	performance_label='accuracy',best_performance=None,
	# 	savename=savename)
	plot_generator.make_plots(fontsize=12,legend_fontsize=8,
		performance_label='J_pi_new',best_performance=None,)



