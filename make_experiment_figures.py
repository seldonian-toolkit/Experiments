# Standard library imports
import argparse
import os
import glob
import pickle

# Third party imports
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.ticker import FormatStrFormatter

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
	parser.add_argument('--savename',
		help='Filename to which the plot will be saved. Full or relative path acceptable. If not provided, plot will not be saved.')
	parser.add_argument('--performance_label',type=str,default='Performance',
		help='The name of the performance measure, e.g. accuracy (for plotting)')
	parser.add_argument('--best_performance',type=float,
		help='The theoretical best performance possible, shown as a horizontal dashed line')
	parser.add_argument('--tot_data_size',type=int,
		help=('The total number of rows (or episodes) in the dataset. '
		'Use if you want horizontal axis to be shown in '
		'terms of points instead of fraction of points'))

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
			constraint_str = pt.constraint_str
			constraint_dict[constraint_name] = {
				'delta':delta,
				'constraint_str':constraint_str}

	constraints = list(constraint_dict.keys())

	subfolders = [os.path.basename(f) for f in os.scandir(args.results_dir) if f.is_dir()]
	subfolders = [x for x in subfolders if x!='resampled_datasets']
	subfolders = [x for x in subfolders if x!='resampled_dataframes']

	all_models = [x.split('_results')[0] for x in subfolders]
	seldonian_models = list(set(all_models).intersection(seldonian_model_set))
	baselines = list(set(all_models).difference(seldonian_model_set))
	print()
	print("Plotting with the following baselines: ", baselines)
	print()
	## BASELINE RESULTS SETUP -- same for all constraints
	baseline_dict = {}
	for baseline in baselines:
		baseline_dict[baseline] = {}
		savename_baseline = os.path.join(
			args.results_dir,f"{baseline}_results",f"{baseline}_results.csv")
		df_baseline = pd.read_csv(savename_baseline)
		n_trials = df_baseline['trial_i'].max()+1

		# Performance
		baseline_mean_performance=df_baseline.groupby('data_pct').mean()['performance']
		baseline_std_performance=df_baseline.groupby('data_pct').std()['performance']
		baseline_ste_performance = baseline_std_performance/np.sqrt(n_trials)
		baseline_dict[baseline]['mean_performance'] = baseline_mean_performance
		baseline_dict[baseline]['ste_performance'] = baseline_ste_performance

	## PLOTTING SETUP
	fig = plt.figure(figsize=(8,4))
	plot_index=1
	n_rows=len(constraints)
	n_cols=3
	fontsize=10
	legend_fontsize=8
	## Loop over constraints and plot baseline and Seldonian results
	for ii,constraint in enumerate(constraints):
		constraint_str = constraint_dict[constraint]['constraint_str']
		delta = constraint_dict[constraint]['delta']

		ax_performance=fig.add_subplot(n_rows,n_cols,plot_index)
		plot_index+=1
		ax_sr=fig.add_subplot(n_rows,n_cols,plot_index)
		plot_index+=1
		ax_fr=fig.add_subplot(n_rows,n_cols,plot_index)
		plot_index+=1

		# Plot labels
		ax_performance.set_ylabel(args.performance_label,fontsize=fontsize)
		ax_sr.set_ylabel('Solution rate')
		ax_fr.set_ylabel('Failure Rate')

		# Only put horizontal axis labels on last row of plots 
		if ii == len(constraints)-1:
			ax_performance.set_xlabel('Training samples',fontsize=fontsize)
			ax_sr.set_xlabel('Training samples',fontsize=fontsize)
			ax_fr.set_xlabel('Training samples',fontsize=fontsize)

		# axis scaling
		ax_performance.set_xscale('log')
		ax_sr.set_xscale('log')
		ax_fr.set_xscale('log')
		
		locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
		locmin = matplotlib.ticker.LogLocator(base=10.0,
			subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
		for ax in [ax_performance,ax_sr,ax_fr]:
			ax.minorticks_on()
			ax.xaxis.set_major_locator(locmaj)
			ax.xaxis.set_minor_locator(locmin)
			ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
		

		# Seldonian results
		filename = os.path.join(
			args.results_dir,
			f"qsa_results",f"qsa_results.csv")

		df_qsa = pd.read_csv(filename)
		passed_mask = df_qsa['passed_safety']==True
		df_qsa_passed = df_qsa[passed_mask]
		
		# performance
		if args.tot_data_size:
			X = df_qsa_passed.groupby('data_pct').mean().index*args.tot_data_size 
			X_all = df_qsa.groupby('data_pct').mean().index*args.tot_data_size 
		else:
			X = df_qsa_passed.groupby('data_pct').mean().index
			X_all = df_qsa.groupby('data_pct').mean().index

		mean_performance = df_qsa_passed.groupby('data_pct').mean()['performance']
		std_performance = df_qsa_passed.groupby('data_pct').std()['performance']
		n_passed = df_qsa_passed.groupby('data_pct').count()['performance']	
		ste_performance = std_performance/np.sqrt(n_passed)

		# Plot baseline performance
		for baseline_i,baseline in enumerate(baselines):
			baseline_color = baseline_colormap(baseline_i)
			baseline_mean_performance = baseline_dict[baseline]['mean_performance']
			
			baseline_ste_performance = baseline_dict[baseline]['ste_performance']
			ax_performance.plot(X_all,baseline_mean_performance,color=baseline_color,label=baseline)
			ax_performance.fill_between(X_all,
				baseline_mean_performance-baseline_ste_performance,
				baseline_mean_performance+baseline_ste_performance,
				color=baseline_color,alpha=0.5)
		# ax_performance.axhline(y=-0.25,color='r',
		# 	linestyle='--',label='random policy')
		# Seldonian 
		ax_performance.plot(X,mean_performance,color='g',
			linestyle='--',label='QSA')
		ax_performance.scatter(X,mean_performance,color='g',s=25)
		ax_performance.fill_between(X,
			mean_performance-ste_performance,
			mean_performance+ste_performance,
			color='g',alpha=0.5)

		ax_performance.set_ylim(0,1.0)

		if args.best_performance:
			ax_performance.plot(X,[args.best_performance for x in X],
				color='k',linestyle='--',label='optimal performance')
		ax_performance.legend(loc='best',fontsize=legend_fontsize)
		
		# ax_performance.tick_params(axis='x', which='minor',length=10,width=2)
		# ax_performance.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

		# Solution rate - calculated on all points
		n_trials = df_qsa['trial_i'].max()+1
		mean_sr = df_qsa.groupby('data_pct').mean()['passed_safety']
		std_sr = df_qsa.groupby('data_pct').std()['passed_safety']
		ste_sr = std_sr/np.sqrt(n_trials)
		
		title =  f'{constraint}: \ng={constraint_str}'
		ax_sr.set_title(title,y=1.05,fontsize=10)

		# Plot baseline solution rate (by default 1.0)
		for baseline_i,baseline in enumerate(baselines):
			baseline_color = baseline_colormap(baseline_i)
			ax_sr.plot(X_all,np.ones_like(X_all),color=baseline_color,label=baseline)

		ax_sr.plot(X_all,mean_sr,color='g',linestyle='--',label='QSA')
		ax_sr.scatter(X_all,mean_sr,color='g',s=25)
		ax_sr.fill_between(X_all,mean_sr-ste_sr,mean_sr+ste_sr,color='g',alpha=0.5)
		# ax_sr.set_ylim(-0.05,1.05)
		
		ax_sr.legend(loc='best',fontsize=legend_fontsize)

		## Failure rate - calculated on all points
		# First baseline
		
		for baseline_i,baseline in enumerate(baselines):
			baseline_color = baseline_colormap(baseline_i)
			# Failure rate
			baseline_mean_fr= df_baseline.groupby('data_pct').mean()['failed']
			baseline_std_fr = df_baseline.groupby('data_pct').std()['failed']
			baseline_ste_fr = baseline_std_fr/np.sqrt(n_trials)	

			ax_fr.plot(X_all,baseline_mean_fr,color=baseline_color,label=baseline)
			ax_fr.fill_between(X_all,baseline_mean_fr-baseline_ste_fr,
				baseline_mean_fr+baseline_ste_fr,
				color=baseline_color,alpha=0.5)
		
		mean_fr=df_qsa.groupby('data_pct').mean()['failed']
		std_fr=df_qsa.groupby('data_pct').std()['failed']
		ste_fr = std_fr/np.sqrt(n_trials)	
		ax_fr.plot(X_all,mean_fr,color='g',linestyle='--',label='QSA')
		ax_fr.fill_between(X_all,
			mean_fr-ste_fr,
			mean_fr+ste_fr,color='g',alpha=0.5)
		ax_fr.scatter(X_all,mean_fr,color='g',s=25)
		ax_fr.axhline(y=delta,color='k',
			linestyle='--',label=f'delta={delta}')
		ax_fr.legend(loc='best',fontsize=legend_fontsize)
		ax_fr.set_ylim(-0.05,1.05)
		# ax_performance.get_xaxis().set_tick_params(which='minor', size=8)

	plt.tight_layout()
	# plt.subplots_adjust(hspace=0.6,wspace=0.3)
	if args.savename:

		plt.savefig(args.savename,format='png',dpi=600)
		print(f"Saved {args.savename}")
	else:
		plt.show()
	

	# # 