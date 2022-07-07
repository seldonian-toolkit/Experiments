""" Utilities used in the rest of the library """

import os
import pickle

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
