import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.generate_plots import RLPlotGenerator

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.environments.gridworld import Gridworld
from experiments.utils import generate_episodes_and_calc_J
	
if __name__ == "__main__":
	# Parameter setup
	run_experiments = True
	make_plots = True
	save_plot = False
	performance_metric = 'J(pi_new)'
	n_trials = 20
	# n_trials = 1
	data_fracs = np.logspace(-2.3,0,10)
	# data_fracs = [0.05]
	n_workers = 8
	verbose=True
	results_dir = f'../../results/gridworld_externalenv_debug'
	os.makedirs(results_dir,exist_ok=True)
	plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
	n_episodes_for_eval = 1000
	# Load spec
	specfile = f'../../../engine-repo-dev/examples/gridworld_externalenv/spec.pkl'
	spec = load_pickle(specfile)
	spec.optimization_hyperparams['num_iters'] = 30
	spec.optimization_hyperparams['alpha_theta'] = 0.01
	spec.optimization_hyperparams['alpha_lamb'] = 0.01
	spec.optimization_hyperparams['beta_velocity'] = 0.9
	spec.optimization_hyperparams['beta_rmspropr'] = 0.95

	perf_eval_fn = generate_episodes_and_calc_J
	perf_eval_kwargs = {
		'n_episodes_for_eval':n_episodes_for_eval
	}

	hyperparameter_and_setting_dict = {}
	hyperparameter_and_setting_dict["env"] = Gridworld()
	hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
	hyperparameter_and_setting_dict["num_episodes"] = 1000
	hyperparameter_and_setting_dict["num_trials"] = 1
	hyperparameter_and_setting_dict["vis"] = False

	plot_generator = RLPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		n_workers=n_workers,
		datagen_method='generate_episodes',
		hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
		perf_eval_fn=perf_eval_fn,
		perf_eval_kwargs=perf_eval_kwargs,
		results_dir=results_dir,
		)
	if run_experiments:
		plot_generator.run_seldonian_experiment(verbose=verbose)

	if make_plots:
		if save_plot:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric,
				savename=plot_savename)
		else:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric,)


