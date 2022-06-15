import os
import glob
import pickle
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt


from seldonian.utils.io_utils import dir_path

from experiments.experiments import (
	BaselineExperiment,SeldonianExperiment)
from experiments.utils import generate_resampled_datasets

seldonian_model_set = set(['qsa','sa'])
baseline_colormap = matplotlib.cm.get_cmap('tab10')

class PlotGenerator():
	""" Class for running Seldonian experiments 
	and generating the three plots:
	1) Performance
	2) Solution rate
	3) Failure rate 
	all plotted vs. amount of data used

	:param spec: Specification object for running the 
		Seldonian algorithm
	:type spec: seldonian.spec.Spec object

	:param n_trials: The number of times the 
		Seldonian algorithm is run for each data fraction.
		Used for generating error bars
	:type n_trials: int

	:param data_pcts: Proportions of the overall size
		of the dataset to use
		(the horizontal axis on the three plots).
	:type data_pcts: List(float)

	:param datagen_method: Method for generating data that is used
		to run the Seldonian algorithm for each trial
	:type datagen_method: str, e.g. "resample"

	:param perf_eval_fn: Function used to evaluate the performance
		of the model obtained in each trial, with signature:
		func(theta,**kwargs), where theta is the solution
		from candidate selection
	:type perf_eval_fn: function or class method


	:param results_dir: The directory in which to save the results
	:type results_dir: str

	:param n_workers: The number of workers to use if
		using multiprocessing
	:type n_workers: int

	:param constraint_eval_fns: List of functions used to evaluate
		the constraints on ground truth. If an empty list is provided,
		the constraints are evaluated using the parse tree 
	:type constraint_eval_fns: List(function or class method), 
		defaults to []
	"""
	def __init__(self,
		spec,
		n_trials,
		data_pcts,
		datagen_method,
		perf_eval_fn,
		results_dir,
		n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs={},
		constraint_eval_kwargs={},
		):
		self.spec = spec
		self.n_trials = n_trials
		self.data_pcts = data_pcts
		self.datagen_method = datagen_method
		self.perf_eval_fn = perf_eval_fn
		self.results_dir = results_dir
		self.n_workers = n_workers
		self.constraint_eval_fns = constraint_eval_fns
		self.perf_eval_kwargs = perf_eval_kwargs
		self.constraint_eval_kwargs = constraint_eval_kwargs

	def make_plots(self,fontsize=12,legend_fontsize=8,
		performance_label='accuracy',best_performance=None,
		savename=None):
		regime = self.spec.dataset.regime
		if regime == 'supervised':
			tot_data_size = len(self.spec.dataset.df)
		elif regime == 'RL':
			tot_data_size = len(self.spec.dataset.episodes)

		parse_trees = self.spec.parse_trees

		constraint_dict = {}
		for pt_ii,pt in enumerate(parse_trees):
			delta = pt.delta
			constraint_str = pt.constraint_str
			constraint_dict[f'constraint_{pt_ii}'] = {
				'delta':delta,
				'constraint_str':constraint_str}

		constraints = list(constraint_dict.keys())
		# print(os.scandir(self.results_dir))
		subfolders = [os.path.basename(f) for f in os.scandir(self.results_dir) if f.is_dir()]
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
				self.results_dir,f"{baseline}_results",f"{baseline}_results.csv")
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
		fontsize=fontsize
		legend_fontsize=legend_fontsize

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
			ax_performance.set_ylabel(performance_label,fontsize=fontsize)
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
				self.results_dir,
				f"qsa_results",f"qsa_results.csv")

			df_qsa = pd.read_csv(filename)
			passed_mask = df_qsa['passed_safety']==True
			df_qsa_passed = df_qsa[passed_mask]
			
			# performance
			if tot_data_size:
				X = df_qsa_passed.groupby('data_pct').mean().index*tot_data_size 
				X_all = df_qsa.groupby('data_pct').mean().index*tot_data_size 
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

			if best_performance:
				ax_performance.plot(X,[best_performance for x in X],
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
			ax_sr.set_ylim(-0.05,1.05)
			
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
		if savename:
			plt.savefig(savename,format='png',dpi=600)
			print(f"Saved {savename}")
		else:
			plt.show()


class SupervisedPlotGenerator(PlotGenerator):
	def __init__(self,
		spec,
		n_trials,
		data_pcts,
		datagen_method,
		perf_eval_fn,
		results_dir,
		n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs={},
		constraint_eval_kwargs={},
		):
		"""Class for running supervised Seldonian experiments 
			and generating the three plots

		:param spec: Specification object for running the 
			Seldonian algorithm
		:type spec: seldonian.spec.Spec object

		:param n_trials: The number of times the 
			Seldonian algorithm is run for each data fraction.
			Used for generating error bars
		:type n_trials: int

		:param data_pcts: Proportions of the overall size
			of the dataset to use
			(the horizontal axis on the three plots).
		:type data_pcts: List(float)

		:param datagen_method: Method for generating data that is used
			to run the Seldonian algorithm for each trial
		:type datagen_method: str, e.g. "resample"

		:param perf_eval_fn: Function used to evaluate the performance
			of the model obtained in each trial, with signature:
			func(theta,**kwargs), where theta is the solution
			from candidate selection
		:type perf_eval_fn: function or class method

		:param results_dir: The directory in which to save the results
		:type results_dir: str

		:param n_workers: The number of workers to use if
			using multiprocessing
		:type n_workers: int

		:param constraint_eval_fns: List of functions used to evaluate
			the constraints on ground truth. If an empty list is provided,
			the constraints are evaluated using the parse tree 
		:type constraint_eval_fns: List(function or class method), 
			defaults to []
		"""

		super().__init__(spec=spec,
			n_trials=n_trials,
			data_pcts=data_pcts,
			datagen_method=datagen_method,
			perf_eval_fn=perf_eval_fn,
			results_dir=results_dir,
			n_workers=n_workers,
			constraint_eval_fns=constraint_eval_fns,
			perf_eval_kwargs=perf_eval_kwargs,
			constraint_eval_kwargs=constraint_eval_kwargs,
			)
		self.regime = 'supervised'

	def run_seldonian_experiment(self,verbose):
		dataset = self.spec.dataset

		label_column = dataset.label_column
		sensitive_column_names = dataset.sensitive_column_names
		include_sensitive_columns = dataset.include_sensitive_columns
		include_intercept_term = dataset.include_intercept_term
		
		if self.datagen_method == 'resample':
			# Generate n_trials resampled datasets of full length
			# These will be cropped to data_pct fractional size
			print("generating resampled datasets")
			generate_resampled_datasets(dataset.df,
				self.n_trials,
				self.results_dir,
				file_format='pkl')

		run_seldonian_kwargs = dict(
			spec=self.spec,
			data_pcts=self.data_pcts,
			n_trials=self.n_trials,
			n_workers=self.n_workers,
			datagen_method=self.datagen_method,
			perf_eval_fn=self.perf_eval_fn,
			perf_eval_kwargs=self.perf_eval_kwargs,
			constraint_eval_fns=self.constraint_eval_fns,
			constraint_eval_kwargs=self.constraint_eval_kwargs,
			verbose=verbose,
			)

		## Run experiment 
		sd_exp = SeldonianExperiment(results_dir=self.results_dir)

		sd_exp.run_experiment(**run_seldonian_kwargs)


class RLPlotGenerator(PlotGenerator):
	def __init__(self,
		spec,
		n_trials,
		data_pcts,
		datagen_method,
		perf_eval_fn,
		RL_environment_obj,
		n_episodes_for_eval,
		results_dir,
		n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs={},
		constraint_eval_kwargs={},
		):
		"""Class for running RL Seldonian experiments 
			and generating the three plots

		:param spec: Specification object for running the 
			Seldonian algorithm
		:type spec: seldonian.spec.Spec object

		:param n_trials: The number of times the 
			Seldonian algorithm is run for each data fraction.
			Used for generating error bars
		:type n_trials: int

		:param data_pcts: Proportions of the overall size
			of the dataset to use
			(the horizontal axis on the three plots).
		:type data_pcts: List(float)

		:param datagen_method: Method for generating data that is used
			to run the Seldonian algorithm for each trial
		:type datagen_method: str, e.g. "resample"

		:param perf_eval_fn: Function used to evaluate the performance
			of the model obtained in each trial, with signature:
			func(theta,**kwargs), where theta is the solution
			from candidate selection
		:type perf_eval_fn: function or class method
		
		:param RL_environment_obj: The RL environment object  
			from the seldonian library 
		:type RL_environment_obj: Environment() class instance

		:param n_episodes_for_eval: The number of episodes to use
			when evaluating the performance.
		:type n_episodes_for_eval: int

		:param results_dir: The directory in which to save the results
		:type results_dir: str

		:param n_workers: The number of workers to use if
			using multiprocessing
		:type n_workers: int

		:param constraint_eval_fns: List of functions used to evaluate
			the constraints on ground truth. If an empty list is provided,
			the constraints are evaluated using the parse tree 
		:type constraint_eval_fns: List(function or class method), 
			defaults to []
		"""

		super().__init__(spec=spec,
			n_trials=n_trials,
			data_pcts=data_pcts,
			datagen_method=datagen_method,
			perf_eval_fn=perf_eval_fn,
			results_dir=results_dir,
			n_workers=n_workers,
			constraint_eval_fns=constraint_eval_fns,
			perf_eval_kwargs=perf_eval_kwargs,
			constraint_eval_kwargs=constraint_eval_kwargs,
			)
		
		self.regime = 'RL'
		self.RL_environment_obj = RL_environment_obj

	def run_seldonian_experiment(self,verbose):
		print("Running experiment")
		dataset = self.spec.dataset
		
		if self.datagen_method == 'generate_episodes':
			# generate full-size datasets for each trial so that 
			# we can reference them for each data_pct
			save_dir = os.path.join(self.results_dir,'resampled_datasets')
			os.makedirs(save_dir,exist_ok=True)
			print("generating resampled datasets")
			for trial_i in range(self.n_trials):
				print(f"Trial: {trial_i}")
				savename = os.path.join(save_dir,f'resampled_data_trial{trial_i}.pkl')
				if not os.path.exists(savename):
					self.RL_environment_obj.generate_data(
						n_episodes=self.perf_eval_kwargs['n_episodes'],
						parallel=True if self.n_workers > 1 else False,
						n_workers=self.n_workers,
						savename=savename)
				else:
					print(f"{savename} already created")

		run_seldonian_kwargs = dict(
			spec=self.spec,
			data_pcts=self.data_pcts,
			n_trials=self.n_trials,
			n_workers=self.n_workers,
			datagen_method=self.datagen_method,
			RL_environment_obj=self.RL_environment_obj,
			constraint_eval_fns=self.constraint_eval_fns,
			constraint_eval_kwargs=self.constraint_eval_kwargs,
			perf_eval_fn=self.perf_eval_fn,
			perf_eval_kwargs=self.perf_eval_kwargs,
			verbose=verbose,
			)


		# ## Run experiment 
		sd_exp = SeldonianExperiment(results_dir=self.results_dir)

		sd_exp.run_experiment(**run_seldonian_kwargs)

		



	
	
