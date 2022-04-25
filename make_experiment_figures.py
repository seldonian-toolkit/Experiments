# Standard library imports
import argparse
import os
import glob
import pickle

# Third party imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Seldonian imports
from seldonian.io_utils import dir_path


seldonian_model_set = set(['qsa','sa'])
baseline_colormap = matplotlib.cm.get_cmap('tab10')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('results_dir',type=dir_path,
		help='The directory where all of your experiment results are saved')
	parser.add_argument('interface_output_pth',
	   type=dir_path, help='Path to output folder from running interface')
	parser.add_argument('plot_save_dir',type=dir_path,
		help='The directory where you want to save the plot')
	parser.add_argument('--tot_data_size',type=int,
		help=('The total number of rows in the dataset.'
		' Use if you want horizontal axis to be shown in'
		' terms of points instead of fraction of points'))

	args = parser.parse_args()

	# Get constraint names, deltas from saved parse trees
	parse_tree_paths = glob.glob(
		os.path.join(args.interface_output_pth,'parse_tree*'))
	constraint_dict = {}
	for pth in parse_tree_paths:
		with open(pth,'rb') as infile:
			constraint_name = os.path.basename(
				pth).split('parse_tree_')[-1].split('.p')[0]
			pt = pickle.load(infile)
			delta = pt.delta
			constraint_dict[constraint_name] = {'delta':delta}

	constraints = list(constraint_dict.keys())

	subfolders = [os.path.basename(f) for f in os.scandir(args.results_dir) if f.is_dir()]

	all_models = [x.split('_results')[0] for x in subfolders]
	seldonian_models = list(set(all_models).intersection(seldonian_model_set))
	baselines = list(set(all_models).difference(seldonian_model_set))
	
	## BASELINE RESULTS SETUP -- same for all constraints
	baseline_dict = {}
	for baseline in baselines:
		baseline_dict[baseline] = {}
		savename_baseline = os.path.join(
			args.results_dir,f"{baseline}_results",f"{baseline}_results.csv")
		df_baseline = pd.read_csv(savename_baseline)
		n_trials = df_baseline['trial_i'].max()+1

		# Accuracy
		baseline_mean_acc=df_baseline.groupby('data_pct').mean()['acc']
		baseline_std_acc=df_baseline.groupby('data_pct').std()['acc']
		baseline_ste_acc = baseline_std_acc/np.sqrt(n_trials)
		baseline_dict[baseline]['mean_acc'] = baseline_mean_acc
		baseline_dict[baseline]['ste_acc'] = baseline_ste_acc


	## PLOTTING SETUP
	fig = plt.figure(figsize=(8,4))
	plot_index=1
	n_rows=len(constraints)
	n_cols=3
	fontsize=10

	## Loop over constraints and plot baseline and Seldonian results
	for ii,constraint in enumerate(constraints):
		if constraint == 'demographic_parity':
			constraint_str = 'abs((PR | [M]) - (PR | [F])) - 0.15'

		delta = constraint_dict[constraint]['delta']
		print(constraint,delta)
		ax_acc=fig.add_subplot(n_rows,n_cols,plot_index)
		plot_index+=1
		ax_sr=fig.add_subplot(n_rows,n_cols,plot_index)
		plot_index+=1
		ax_fr=fig.add_subplot(n_rows,n_cols,plot_index)
		plot_index+=1

		# Plot labels
		ax_acc.set_ylabel('Accuracy',fontsize=fontsize)
		ax_sr.set_ylabel('Solution rate')
		ax_fr.set_ylabel('Failure Rate')

		# Only put horizontal axis labels on last row of plots 
		if ii == len(constraints)-1:
			ax_acc.set_xlabel('Training samples',fontsize=fontsize)
			ax_sr.set_xlabel('Training samples',fontsize=fontsize)
			ax_fr.set_xlabel('Training samples',fontsize=fontsize)

		# axis scaling
		ax_acc.set_xscale('log')
		ax_sr.set_xscale('log')
		ax_fr.set_xscale('log')

		# Seldonian results
		filename = os.path.join(
			args.results_dir,
			f"qsa_results",f"qsa_results.csv")

		df_qsa = pd.read_csv(filename)
		passed_mask = df_qsa['passed_safety']==True
		df_qsa_passed = df_qsa[passed_mask]
		
		# accuracy 
		if args.tot_data_size:
			X = df_qsa_passed.groupby('data_pct').mean().index*args.tot_data_size 
		else:
			X = df_qsa_passed.groupby('data_pct').mean().index

		mean_acc = df_qsa_passed.groupby('data_pct').mean()['acc']
		std_acc = df_qsa_passed.groupby('data_pct').std()['acc']
		n_passed = df_qsa_passed.groupby('data_pct').count()['acc']	
		ste_acc = std_acc/np.sqrt(n_passed)

		# Plot baseline accuracy
		for baseline_i,baseline in enumerate(baselines):
			baseline_color = baseline_colormap(baseline_i)
			baseline_mean_acc = baseline_dict[baseline]['mean_acc']
			baseline_ste_acc = baseline_dict[baseline]['ste_acc']
			ax_acc.plot(X,baseline_mean_acc,color=baseline_color)
			ax_acc.fill_between(X,
				baseline_mean_acc-baseline_ste_acc,
				baseline_mean_acc+baseline_ste_acc,
				color=baseline_color,alpha=0.5)
		
		# Seldonian 
		ax_acc.plot(X,mean_acc,color='g',linestyle='--',)
		ax_acc.fill_between(X,
			mean_acc-ste_acc,
			mean_acc+ste_acc,
			color='g',alpha=0.5)

		# Solution rate - calculated on all points
		mean_sr = df_qsa.groupby('data_pct').mean()['passed_safety']
		std_sr = df_qsa.groupby('data_pct').std()['passed_safety']
		ste_sr = std_sr/np.sqrt(n_trials)
		
		title = constraint + ': ' + rf'g=${constraint_str}$'
		ax_sr.set_title(title,y=1.05,fontsize=10)

		# Plot baseline solution rate (by default 1.0)
		for baseline_i,baseline in enumerate(baselines):
			baseline_color = baseline_colormap(baseline_i)
			ax_sr.plot(X,np.ones_like(X),color=baseline_color,label=baseline)

		ax_sr.plot(X,mean_sr,color='g',linestyle='--',label='QSA')
		ax_sr.fill_between(X,mean_sr-ste_sr,mean_sr+ste_sr,color='g',alpha=0.5)
		ax_sr.set_ylim(-0.05,1.05)
		
		ax_sr.legend(loc='lower center',fontsize=7)

		## Failure rate - calculated on all points
		# First baseline
		
		for baseline_i,baseline in enumerate(baselines):
			baseline_color = baseline_colormap(baseline_i)
			# Failure rate
			baseline_mean_fr= df_baseline.groupby('data_pct').mean()['failed']
			baseline_std_fr = df_baseline.groupby('data_pct').std()['failed']
			baseline_ste_fr = baseline_std_fr/np.sqrt(n_trials)	

			ax_fr.plot(X,baseline_mean_fr,color=baseline_color)
			ax_fr.fill_between(X,baseline_mean_fr-baseline_ste_fr,
				baseline_mean_fr+baseline_ste_fr,
				color=baseline_color,alpha=0.5)
		
		mean_fr=df_qsa.groupby('data_pct').mean()['failed']
		std_fr=df_qsa.groupby('data_pct').std()['failed']
		ste_fr = std_fr/np.sqrt(n_trials)	
		ax_fr.plot(X,mean_fr,color='g',linestyle='--')
		ax_fr.fill_between(X,
			mean_fr-ste_fr,
			mean_fr+ste_fr,color='g',alpha=0.5)
		ax_fr.axhline(y=delta,color='k',linestyle='--')
		ax_fr.set_ylim(-0.05,1.05)


	plt.tight_layout()
	# plt.subplots_adjust(hspace=0.6,wspace=0.3)
	plt.show()
	# # savename = os.path.join(plot_dir,'science_gpa_classification_v4.png')
	# # plt.savefig(savename,format='png',dpi=600)
	# # print(f"Saved {savename}")

	# # 