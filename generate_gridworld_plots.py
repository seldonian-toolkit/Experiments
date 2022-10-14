import os
import autograd.numpy as np   # Thinly-wrapped version of Numpy

from experiments.generate_plots import RLPlotGenerator

from seldonian.utils.io_utils import load_pickle
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.RL.RL_runner import (create_env,
	create_agent,run_trial_given_agent_and_env)

def generate_episodes_and_calc_J(**kwargs):
    """ Calculate the expected discounted return 
    by generating episodes

    :return: episodes, J, where episodes is the list
        of generated ground truth episodes and J is
        the expected discounted return
    :rtype: (List(Episode),float)
    """
    # Get trained model weights from running the Seldonian algo
    model = kwargs['model']
    new_params = model.policy.get_params()
   
    # create env and agent
    hyperparameter_and_setting_dict = kwargs['hyperparameter_and_setting_dict']
    agent = create_agent(hyperparameter_and_setting_dict)
    env = create_env(hyperparameter_and_setting_dict)
   
    # set agent's weights to the trained model weights
    agent.set_new_params(new_params)
    
    # generate episodes
    num_episodes = kwargs['n_episodes_for_eval']
    episodes = run_trial_given_agent_and_env(
        agent=agent,env=env,num_episodes=num_episodes)

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards,env.gamma) for ep in episodes])
    J = np.mean(returns)
    return episodes,J
	
if __name__ == "__main__":
	# Parameter setup
	run_experiments = True
	make_plots = True
	save_plot = False
	performance_metric = 'J(pi_new)'
	# n_trials = 20
	n_trials = 10
	data_fracs = np.logspace(-2.3,0,10)
	# data_fracs = [0.01]
	n_workers = 7
	verbose=True
	results_dir = f'results/gridworld_2022Oct14_debug'
	os.makedirs(results_dir,exist_ok=True)
	plot_savename = os.path.join(results_dir,f'gridworld_{n_trials}trials.png')
	n_episodes_for_eval = 1000
	# Load spec
	specfile = f'../engine-repo-dev/examples/gridworld_tutorial/spec.pkl'
	spec = load_pickle(specfile)
	spec.optimization_hyperparams['num_iters'] = 40
	spec.optimization_hyperparams['alpha_theta'] = 0.01
	spec.optimization_hyperparams['alpha_lamb'] = 0.01
	spec.optimization_hyperparams['beta_velocity'] = 0.9
	spec.optimization_hyperparams['beta_rmspropr'] = 0.95

	perf_eval_fn = generate_episodes_and_calc_J
	perf_eval_kwargs = {
		'n_episodes_for_eval':n_episodes_for_eval
	}

	hyperparameter_and_setting_dict = {}
	hyperparameter_and_setting_dict["env"] = "gridworld"
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


