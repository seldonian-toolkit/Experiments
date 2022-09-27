""" Utilities used in the rest of the library """

import os
import pickle
import numpy as np

from seldonian.RL.RL_runner import (create_env,
	create_agent,run_trial_given_agent_and_env)
from seldonian.utils.stats_utils import weighted_sum_gamma


def generate_resampled_datasets(df,n_trials,save_dir,file_format='csv'):
	"""Utility function for supervised learning to generate the
	resampled datasets to use in each trial. Resamples (with replacement)
	pandas dataframes to create n_trials resampled dataframes of the same
	shape as the input dataframe, df

	:param df: The original df from which to resample 
	:type df: pandas DataFrame

	:param n_trials: The number of trials, i.e. the number of 
		resampled datasets to make
	:type n_trials: int

	:param save_dir: The parent directory in which to save the 
		resampled datasets
	:type save_dir: str

	:param file_format: The format of the saved datasets, options are 
		"csv" and "pkl"
	:type file_format: str

	"""
	save_subdir = os.path.join(save_dir,
			'resampled_dataframes')
	os.makedirs(save_subdir,exist_ok=True)

	for trial_i in range(n_trials):

		savename = os.path.join(save_subdir,
			f'trial_{trial_i}.{file_format}')

		if not os.path.exists(savename):
			resampled_df = df.sample(
				n=len(df),replace=True).reset_index(drop=True)

			if file_format == 'csv':
				resampled_df.to_csv(savename,index=False)

			elif file_format == 'pkl':
				with open(savename,'wb') as outfile:
					pickle.dump(resampled_df,outfile)
			print(f"Saved {savename}")	

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
	


def MSE(y_pred,y,**kwargs):
	""" Calculate sample mean squared error 

	:param y_pred: Array of predicted labels
	:param y: Array of true labels
	"""
	n = len(y)
	res = sum(pow(y_pred-y,2))/n
	return res