import os
import pickle

def generate_resampled_datasets(df,n_trials,save_dir,file_format='csv'):
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
