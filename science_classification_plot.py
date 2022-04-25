import os
import glob
import time
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from seldonian.dataset import *
from seldonian.model import *
from seldonian.candidate_selection import CandidateSelection
from seldonian.safety_test import SafetyTest
from seldonian.parse_tree import ParseTree
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

plot_dir = './classification_plots'
n_trials = 250
data_pcts = [0.005,0.01, 0.012742749857, 0.0162377673919,
	 0.0206913808111, 0.0263665089873, 0.0335981828628,
	 0.0428133239872, 0.0545559478117, 0.0695192796178,
	 0.088586679041, 0.112883789168, 0.143844988829,
	 0.183298071083, 0.233572146909, 0.297635144163,
	 0.379269019073, 0.483293023857, 0.615848211066,
	 0.784759970351, 1.0]
n_points = 43303
baseline_colors = ['red','orange']
baselines = ['logistic_regression','sgd']
constraints = ['disparate_impact','demographic_parity',
		'equal_opportunity','equalized_odds','predictive_equality']

if __name__ == "__main__":
	start = time.time()

	X=[x*n_points for x in data_pcts]

	
	baseline_dict = {}
	for baseline in baselines:
		baseline_dict[baseline] = {}
		savename_baseline = os.path.join(
			plot_dir,f"results_{baseline}.csv")
		df_baseline = pd.read_csv(savename_baseline)
		
		# Accuracy
		baseline_mean_acc=df_baseline.groupby('data_pct').mean()['accuracy']
		baseline_std_acc=df_baseline.groupby('data_pct').std()['accuracy']
		baseline_ste_acc = baseline_std_acc/np.sqrt(n_trials)
		baseline_dict[baseline]['mean_acc'] = baseline_mean_acc
		baseline_dict[baseline]['ste_acc'] = baseline_ste_acc
		# DI Failure rate
		baseline_mean_fr_disparate_impact=df_baseline.groupby('data_pct').mean()['failed_di']
		baseline_std_fr_disparate_impact=df_baseline.groupby('data_pct').std()['failed_di']
		baseline_ste_fr_disparate_impact = baseline_std_fr_disparate_impact/np.sqrt(n_trials)	
		baseline_dict[baseline]['mean_fr_disparate_impact'] = baseline_mean_fr_disparate_impact
		baseline_dict[baseline]['ste_fr_disparate_impact'] = baseline_ste_fr_disparate_impact
		# DP Failure rate
		baseline_mean_fr_demographic_parity= df_baseline.groupby('data_pct').mean()['failed_dp']
		baseline_std_fr_demographic_parity = df_baseline.groupby('data_pct').std()['failed_dp']
		baseline_ste_fr_demographic_parity = baseline_std_fr_demographic_parity/np.sqrt(n_trials)	
		baseline_dict[baseline]['mean_fr_demographic_parity'] = baseline_mean_fr_demographic_parity
		baseline_dict[baseline]['ste_fr_demographic_parity'] = baseline_ste_fr_demographic_parity
		# EO Failure rate
		baseline_mean_fr_equal_opportunity=df_baseline.groupby('data_pct').mean()['failed_eo']
		baseline_std_fr_equal_opportunity=df_baseline.groupby('data_pct').std()['failed_eo']
		baseline_ste_fr_equal_opportunity = baseline_std_fr_equal_opportunity/np.sqrt(n_trials)
		baseline_dict[baseline]['mean_fr_equal_opportunity'] = baseline_mean_fr_equal_opportunity
		baseline_dict[baseline]['ste_fr_equal_opportunity'] = baseline_ste_fr_equal_opportunity
		# EODDS Failure rate
		baseline_mean_fr_equalized_odds=df_baseline.groupby('data_pct').mean()['failed_eodds']
		baseline_std_fr_equalized_odds=df_baseline.groupby('data_pct').std()['failed_eodds']
		baseline_ste_fr_equalized_odds = baseline_std_fr_equalized_odds/np.sqrt(n_trials)
		baseline_dict[baseline]['mean_fr_equalized_odds'] = baseline_mean_fr_equalized_odds
		baseline_dict[baseline]['ste_fr_equalized_odds'] = baseline_ste_fr_equalized_odds
		# EODDS_Stephen Failure rate
		baseline_mean_fr_equalized_odds_stephen=df_baseline.groupby('data_pct').mean()['failed_eodds_stephen']
		baseline_std_fr_equalized_odds_stephen=df_baseline.groupby('data_pct').std()['failed_eodds_stephen']
		baseline_ste_fr_equalized_odds_stephen = baseline_std_fr_equalized_odds_stephen/np.sqrt(n_trials)	
		baseline_dict[baseline]['mean_fr_equalized_odds_stephen'] = baseline_mean_fr_equalized_odds_stephen
		baseline_dict[baseline]['ste_fr_equalized_odds_stephen'] = baseline_ste_fr_equalized_odds_stephen
		# Predictive equality
		baseline_mean_fr_predictive_equality = df_baseline.groupby('data_pct').mean()['failed_pe']
		baseline_std_fr_predictive_equality  = df_baseline.groupby('data_pct').std()['failed_pe']
		baseline_ste_fr_predictive_equality  = baseline_std_fr_predictive_equality/np.sqrt(n_trials)		
		baseline_dict[baseline]['mean_fr_predictive_equality'] = baseline_mean_fr_predictive_equality
		baseline_dict[baseline]['ste_fr_predictive_equality'] = baseline_ste_fr_predictive_equality
	# Loop over constraints, calculate statistics, and plot
	fig = plt.figure(figsize=(8.5,8))
	plot_index = 1
	n_rows = len(constraints)
	n_cols = 3
	fontsize=10
	for ii,constraint in enumerate(constraints):
		if constraint == 'disparate_impact':
			constraint_str = '0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'
		elif constraint == 'demographic_parity':
			constraint_str = 'abs((PR | [M]) - (PR | [F])) - 0.15'
		elif constraint == 'equal_opportunity':
			constraint_str = 'abs((FNR | [M]) - (FNR | [F])) - 0.2'
		elif constraint == 'equalized_odds':
			constraint_str = 'abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) - 0.35'
		elif constraint == 'equalized_odds_stephen':
			constraint_str = 'abs((TPR | [M]) - (TPR | [F])) + abs((FPR | [M]) - (FPR | [F])) - 0.35'
		elif constraint == 'predictive_equality':
			constraint_str = 'abs((FPR | [M]) - (FPR | [F])) - 0.2'
		ax_acc=fig.add_subplot(n_rows,n_cols,plot_index)
		
		filename = os.path.join(
			plot_dir,f"results_{constraint}.csv")

		df = pd.read_csv(filename)
		passed_mask = df['passed_safety']==True
		df_passed = df[passed_mask]
		
		# accuracy 
		data_pcts = df_passed.groupby('data_pct').mean().index*n_points 
		mean_acc = df_passed.groupby('data_pct').mean()['accuracy']
		std_acc = df_passed.groupby('data_pct').std()['accuracy']
		n_passed = df_passed.groupby('data_pct').count()['accuracy']	
		ste_acc = std_acc/np.sqrt(n_passed)

		for baseline_i,baseline in enumerate(baselines):
			baseline_mean_acc = baseline_dict[baseline]['mean_acc']
			baseline_ste_acc = baseline_dict[baseline]['ste_acc']
			ax_acc.plot(X,baseline_mean_acc,color=baseline_colors[baseline_i])
			ax_acc.fill_between(X,
				baseline_mean_acc-baseline_ste_acc,
				baseline_mean_acc+baseline_ste_acc,
				color=baseline_colors[baseline_i],alpha=0.5)
		
		ax_acc.plot(data_pcts,mean_acc,color='g',linestyle='--',)
		ax_acc.fill_between(data_pcts,
			mean_acc-ste_acc,
			mean_acc+ste_acc,
			color='g',alpha=0.5)
		ax_acc.set_ylim(0.35,0.7)
		ax_acc.set_xscale('log')
		ax_acc.set_ylabel('Accuracy',fontsize=fontsize)
		if ii == len(constraints)-1:
			ax_acc.set_xlabel('Training samples',fontsize=fontsize)

		# solution rate - calculated on all points
		mean_sr = df.groupby('data_pct').mean()['passed_safety']
		std_sr = df.groupby('data_pct').std()['passed_safety']
		ste_sr = std_sr/np.sqrt(n_trials)

		plot_index+=1
		ax_sr=fig.add_subplot(n_rows,n_cols,plot_index)
		title = constraint + ': ' + rf'g=${constraint_str}$'
		ax_sr.set_title(title,y=1.05,fontsize=10)
		ax_sr.plot(X,np.ones_like(X),color=baseline_colors[0],label='Logistic Regression')
		ax_sr.plot(X,np.ones_like(X),color=baseline_colors[1],label='SGD')
		ax_sr.plot(X,mean_sr,color='g',linestyle='--',label='QSA Classification')
		ax_sr.fill_between(X,mean_sr-ste_sr,mean_sr+ste_sr,color='g',alpha=0.5)
		ax_sr.set_ylim(-0.05,1.15)
		ax_sr.set_xscale('log')
		ax_sr.set_ylabel('Solution rate')
		if ii == len(constraints)-1:
			ax_sr.set_xlabel('Training samples',fontsize=fontsize)

		# Failure rate - calculated on all points
		mean_fr=df.groupby('data_pct').mean()['failed']
		std_fr=df.groupby('data_pct').std()['failed']
		ste_fr = std_fr/np.sqrt(n_trials)	

		plot_index+=1
		ax_fr=fig.add_subplot(n_rows,n_cols,plot_index)
		
		for baseline_i,baseline in enumerate(baselines):
			baseline_mean_fr = baseline_dict[baseline][f'mean_fr_{constraint}']
			baseline_ste_fr = baseline_dict[baseline][f'ste_fr_{constraint}']
			ax_fr.plot(X,baseline_mean_fr,color=baseline_colors[baseline_i])
			ax_fr.fill_between(X,baseline_mean_fr-baseline_ste_fr,
				baseline_mean_fr+baseline_ste_fr,
				color=baseline_colors[baseline_i],alpha=0.5)

		ax_fr.plot(X,mean_fr,color='g',linestyle='--')
		ax_fr.fill_between(X,
			mean_fr-ste_fr,
			mean_fr+ste_fr,color='g',alpha=0.5)
		ax_fr.axhline(y=0.05,color='k',linestyle='--')
		ylims_fr = ax_fr.get_ylim()
		ymax = ylims_fr[1]
		if ymax < 0.15:
			ymax = 0.15
		ax_fr.set_ylim(-0.05,ymax)
		ax_fr.set_xscale('log')
		ax_fr.set_ylabel('Failure Rate')
		if ii == len(constraints)-1:
			ax_fr.set_xlabel('Training samples',fontsize=fontsize)
		if ii == 0:
			ax_sr.legend(loc='lower center',fontsize=7)
		plot_index+=1

	# plt.tight_layout()
	plt.subplots_adjust(hspace=0.6,wspace=0.3)
	plt.show()
	# savename = os.path.join(plot_dir,'science_gpa_classification_v4.png')
	# plt.savefig(savename,format='png',dpi=600)
	# print(f"Saved {savename}")

	