import numpy as np

# def precalc(dataset):
# 	d = {}
# 	d['expected'] = dataset.df['M']==1
# 	return d

def main_reward_mean(param_weights,**kwargs):	
	# Generate n_episodes many times using candidate policy
	# then take average J and return const - J(pi_c)

	RL_environment_obj = kwargs['RL_environment_obj']

	df_regen = RL_environment_obj.generate_data(
		n_episodes=kwargs['n_episodes_for_eval'])
	J_pi_c = RL_environment_obj.calc_J_from_df(df_regen,
		gamma=RL_environment_obj.gamma)
	
	return -0.25 - J_pi_c

constraints = [main_reward_mean]