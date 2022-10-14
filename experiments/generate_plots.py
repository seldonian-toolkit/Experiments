""" Module for making the three plots """

import os
import glob
import pickle
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

from seldonian.utils.io_utils import save_pickle

from .experiments import (
	BaselineExperiment,SeldonianExperiment,
	FairlearnExperiment)
from .utils import generate_resampled_datasets

seldonian_model_set = set(['qsa','sa'])
plot_colormap = matplotlib.cm.get_cmap('tab10')
marker_list = ['s','p','d','*','x','h','+']

class PlotGenerator():
	def __init__(self,
		spec,
		n_trials,
		data_fracs,
		datagen_method,
		perf_eval_fn,
		results_dir,
		n_workers,
		constraint_eval_fns=[],
		perf_eval_kwargs={},
		constraint_eval_kwargs={},
		):
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

		:param data_fracs: Proportions of the overall size
			of the dataset to use
			(the horizontal axis on the three plots).
		:type data_fracs: List(float)

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

		:param perf_eval_kwargs: Extra keyword arguments to pass to
			perf_eval_fn
		:type perf_eval_kwargs: dict

		:param constraint_eval_kwargs: Extra keyword arguments to pass to
			the constraint_eval_fns
		:type constraint_eval_kwargs: dict
		"""
		self.spec = spec
		self.n_trials = n_trials
		self.data_fracs = data_fracs
		self.datagen_method = datagen_method
		self.perf_eval_fn = perf_eval_fn
		self.results_dir = results_dir
		self.n_workers = n_workers
		self.constraint_eval_fns = constraint_eval_fns
		self.perf_eval_kwargs = perf_eval_kwargs
		self.constraint_eval_kwargs = constraint_eval_kwargs

	def make_plots(self,fontsize=12,legend_fontsize=8,
		performance_label='accuracy',
		marker_size=20,
		include_legend=True,
		savename=None):
		""" Make the three plots from results files saved to
		self.results_dir
		
		:param fontsize: The font size to use for the axis labels
		:type fontsize: int

		:param legend_fontsize: The font size to use for text 
			in the legend
		:type legend_fontsize: int

		:param performance_label: The y axis label on the performance
			plot you want to use. 
		:type performance_label: str, defaults to "accuracy"

		:param savename: If not None, the filename path to which the plot 
			will be saved on disk. 
		:type savename: str, defaults to None
		"""
		regime = self.spec.dataset.regime
		
		if regime == 'supervised_learning':
			tot_data_size = len(self.spec.dataset.df)
		elif regime == 'reinforcement_learning':
			tot_data_size = self.hyperparameter_and_setting_dict['num_episodes']
		else:
			raise NotImplementedError(f"regime={regime} not supported.")
		# Read in constraints
		parse_trees = self.spec.parse_trees

		constraint_dict = {}
		for pt_ii,pt in enumerate(parse_trees):
			delta = pt.delta
			constraint_str = pt.constraint_str
			constraint_dict[f'constraint_{pt_ii}'] = {
				'delta':delta,
				'constraint_str':constraint_str}

		constraints = list(constraint_dict.keys())

		# Figure out what experiments we have from subfolders in results_dir
		subfolders = [os.path.basename(f) for f in os.scandir(self.results_dir) if f.is_dir()]
		all_models = [x.split('_results')[0] for x in subfolders if x.endswith('_results')]
		seldonian_models = list(set(all_models).intersection(seldonian_model_set))
		baselines = sorted(list(set(all_models).difference(seldonian_model_set)))

		if not (seldonian_models or baselines):
			print("No results for Seldonian models or baselines found ")
			return
		
		## BASELINE RESULTS SETUP 
		baseline_dict = {}
		for baseline in baselines:
			baseline_dict[baseline] = {}
			savename_baseline = os.path.join(
				self.results_dir,f"{baseline}_results",f"{baseline}_results.csv")
			df_baseline = pd.read_csv(savename_baseline)
			df_baseline['solution_returned']=df_baseline['performance'].apply(lambda x: ~np.isnan(x))

			valid_mask = ~np.isnan(df_baseline['performance'])
			df_baseline_valid = df_baseline[valid_mask]
			# Get the list of all data_fracs 
			X_all = df_baseline.groupby('data_frac').mean().index*tot_data_size 
			# Get the list of data_fracs for which there is at least one trial that has non-nan performance
			X_valid = df_baseline_valid.groupby('data_frac').mean().index*tot_data_size 

			baseline_dict[baseline]['df_baseline'] = df_baseline.copy()
			baseline_dict[baseline]['df_baseline_valid'] = df_baseline_valid.copy()
			baseline_dict[baseline]['X_all'] = X_all
			baseline_dict[baseline]['X_valid'] = X_valid

		# SELDONIAN RESULTS SETUP
		seldonian_dict = {}
		for seldonian_model in seldonian_models:
			seldonian_dict[seldonian_model] = {}
			savename_seldonian = os.path.join(
				self.results_dir,
				f"{seldonian_model}_results",
				f"{seldonian_model}_results.csv")

			df_seldonian = pd.read_csv(savename_seldonian)
			passed_mask = df_seldonian['passed_safety']==True
			df_seldonian_passed = df_seldonian[passed_mask]
			# Get the list of all data_fracs 
			X_all = df_seldonian.groupby('data_frac').mean().index*tot_data_size 
			# Get the list of data_fracs for which there is at least one trial that passed the safety test
			X_passed = df_seldonian_passed.groupby('data_frac').mean().index*tot_data_size 
			seldonian_dict[seldonian_model]['df_seldonian'] = df_seldonian.copy()
			seldonian_dict[seldonian_model]['df_seldonian_passed'] = df_seldonian_passed.copy()
			seldonian_dict[seldonian_model]['X_all'] = X_all
			seldonian_dict[seldonian_model]['X_passed'] = X_passed

		## PLOTTING SETUP
		if include_legend:
			figsize=(9,4.5)
		else:
			figsize=(9,4)
		fig = plt.figure(figsize=figsize)
		plot_index=1
		n_rows=len(constraints)
		n_cols=3
		fontsize=fontsize
		legend_fontsize=legend_fontsize
		legend_handles = []
		legend_labels = []
		## Loop over constraints and make three plots for each constraint
		for ii,constraint in enumerate(constraints):
			constraint_str = constraint_dict[constraint]['constraint_str']
			delta = constraint_dict[constraint]['delta']

			# SETUP FOR PLOTTING
			ax_performance=fig.add_subplot(n_rows,n_cols,plot_index)
			plot_index+=1
			ax_sr=fig.add_subplot(n_rows,n_cols,plot_index,sharex=ax_performance)
			plot_index+=1
			ax_fr=fig.add_subplot(n_rows,n_cols,plot_index,sharex=ax_performance)
			plot_index+=1

			# Plot title (put above middle plot)
			title =  f'constraint: \ng={constraint_str}'
			ax_sr.set_title(title,y=1.05,fontsize=10)

			# Plot labels
			ax_performance.set_ylabel(performance_label,fontsize=fontsize)
			ax_sr.set_ylabel('Solution rate',fontsize=fontsize)
			ax_fr.set_ylabel('Failure Rate',fontsize=fontsize)

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
			
			########################
			### PERFORMANCE PLOT ###
			########################

			# Baseline performance
			for baseline_i,baseline in enumerate(baselines):
				baseline_color = plot_colormap(baseline_i+len(seldonian_models)) # 0 is reserved for Seldonian model
				this_baseline_dict = baseline_dict[baseline]
				df_baseline_valid = this_baseline_dict['df_baseline_valid']
				n_trials = df_baseline_valid['trial_i'].max()+1

				# Performance
				baseline_mean_performance=df_baseline_valid.groupby('data_frac').mean()['performance']
				baseline_std_performance=df_baseline_valid.groupby('data_frac').std()['performance']
				baseline_ste_performance = baseline_std_performance/np.sqrt(n_trials)
				X_valid_baseline = this_baseline_dict['X_valid']
				pl, = ax_performance.plot(X_valid_baseline,baseline_mean_performance,
					color=baseline_color,label=baseline)
				legend_handles.append(pl)
				legend_labels.append(baseline)
				ax_performance.scatter(X_valid_baseline,baseline_mean_performance,
					color=baseline_color,s=marker_size,
					marker=marker_list[baseline_i])
				ax_performance.fill_between(X_valid_baseline,
					baseline_mean_performance-baseline_ste_performance,
					baseline_mean_performance+baseline_ste_performance,
					color=baseline_color,alpha=0.5)

			for seldonian_i,seldonian_model in enumerate(seldonian_models):
				this_seldonian_dict = seldonian_dict[seldonian_model]
				seldonian_color = plot_colormap(seldonian_i)
				df_seldonian_passed = this_seldonian_dict['df_seldonian_passed']
				mean_performance = df_seldonian_passed.groupby('data_frac').mean()['performance']
				std_performance = df_seldonian_passed.groupby('data_frac').std()['performance']
				n_passed = df_seldonian_passed.groupby('data_frac').count()['performance']	
				ste_performance = std_performance/np.sqrt(n_passed)
				X_passed_seldonian = this_seldonian_dict['X_passed']
				pl, = ax_performance.plot(X_passed_seldonian,mean_performance,color=seldonian_color,
					linestyle='--',label='QSA')
				legend_handles.append(pl)
				legend_labels.append(seldonian_model)
				ax_performance.scatter(X_passed_seldonian,mean_performance,color=seldonian_color,
					s=marker_size,marker='o')
				ax_performance.fill_between(X_passed_seldonian,
					mean_performance-ste_performance,
					mean_performance+ste_performance,
					color=seldonian_color,alpha=0.5)

			##########################
			### SOLUTION RATE PLOT ###
			##########################
			
			# Plot baseline solution rate 
			# (sometimes it doesn't return a solution due to not having enough training data
			# to run model.fit() )
			for baseline_i,baseline in enumerate(baselines):
				this_baseline_dict = baseline_dict[baseline]
				X_all_baseline = this_baseline_dict['X_all']
				baseline_color = plot_colormap(baseline_i+len(seldonian_models))
				df_baseline = this_baseline_dict['df_baseline']
				n_trials = df_baseline['trial_i'].max()+1
				mean_sr = df_baseline.groupby('data_frac').mean()['solution_returned']
				std_sr = df_baseline.groupby('data_frac').std()['solution_returned']
				ste_sr = std_sr/np.sqrt(n_trials)

				X_all_baseline = this_baseline_dict['X_all']

				ax_sr.plot(X_all_baseline,mean_sr,color=baseline_color,label=baseline)
				ax_sr.scatter(X_all_baseline,mean_sr,color=baseline_color,
					s=marker_size,marker=marker_list[baseline_i])
				ax_sr.fill_between(X_all_baseline,mean_sr-ste_sr,mean_sr+ste_sr,
					color=baseline_color,alpha=0.5)

			for seldonian_i,seldonian_model in enumerate(seldonian_models):
				this_seldonian_dict = seldonian_dict[seldonian_model]
				seldonian_color = plot_colormap(seldonian_i)
				df_seldonian = this_seldonian_dict['df_seldonian']
				n_trials = df_seldonian['trial_i'].max()+1
				mean_sr = df_seldonian.groupby('data_frac').mean()['passed_safety']
				std_sr = df_seldonian.groupby('data_frac').std()['passed_safety']
				ste_sr = std_sr/np.sqrt(n_trials)

				X_all_seldonian = this_seldonian_dict['X_all']

				ax_sr.plot(X_all_seldonian,mean_sr,color=seldonian_color,linestyle='--',label='QSA')
				ax_sr.scatter(X_all_seldonian,mean_sr,color=seldonian_color,
					s=marker_size,marker='o')
				ax_sr.fill_between(X_all_seldonian,mean_sr-ste_sr,mean_sr+ste_sr,
					color=seldonian_color,alpha=0.5)
			
			ax_sr.set_ylim(-0.05,1.05)

			##########################
			### FAILURE RATE PLOT ###
			##########################

			# Baseline failure rate
			for baseline_i,baseline in enumerate(baselines):
				baseline_color = plot_colormap(baseline_i+len(seldonian_models))
				# Baseline performance
				this_baseline_dict = baseline_dict[baseline]
				df_baseline_valid = this_baseline_dict['df_baseline_valid']
				n_trials = df_baseline_valid['trial_i'].max()+1

				baseline_mean_fr = df_baseline_valid.groupby('data_frac').mean()['failed']
				baseline_std_fr = df_baseline_valid.groupby('data_frac').std()['failed']
				baseline_ste_fr = baseline_std_fr/np.sqrt(n_trials)
				
				X_valid_baseline = this_baseline_dict['X_valid']


				ax_fr.plot(X_valid_baseline,baseline_mean_fr,
					color=baseline_color,label=baseline)
				ax_fr.scatter(X_valid_baseline,baseline_mean_fr,
					color=baseline_color,marker=marker_list[baseline_i],
					s=marker_size)
				ax_fr.fill_between(X_valid_baseline,baseline_mean_fr-baseline_ste_fr,
					baseline_mean_fr+baseline_ste_fr,
					color=baseline_color,alpha=0.5)

			for seldonian_i,seldonian_model in enumerate(seldonian_models):
				this_seldonian_dict = seldonian_dict[seldonian_model]
				seldonian_color = plot_colormap(seldonian_i)
				df_seldonian = this_seldonian_dict['df_seldonian']
				n_trials = df_seldonian['trial_i'].max()+1
				mean_fr=df_seldonian.groupby('data_frac').mean()['failed']
				std_fr=df_seldonian.groupby('data_frac').std()['failed']
				ste_fr = std_fr/np.sqrt(n_trials)	

				X_all_seldonian = this_seldonian_dict['X_all']
				ax_fr.plot(X_all_seldonian,mean_fr,color=seldonian_color,linestyle='--',label='QSA')
				ax_fr.fill_between(X_all_seldonian,
					mean_fr-ste_fr,
					mean_fr+ste_fr,color=seldonian_color,alpha=0.5)
				ax_fr.scatter(X_all_seldonian,mean_fr,color=seldonian_color,s=marker_size,marker='o')
				ax_fr.axhline(y=delta,color='k',
					linestyle='--',label=f'delta={delta}')
			ax_fr.set_ylim(-0.05,1.05)
		
		plt.tight_layout()
		
		if include_legend:
			fig.subplots_adjust(bottom=0.25)
			ncol = 4
			fig.legend(legend_handles,legend_labels,
				bbox_to_anchor=(0.5,0.15),loc="upper center",ncol=ncol)
		
		if savename:
			plt.savefig(savename,format='png',dpi=600)
			print(f"Saved {savename}")
		else:
			plt.show()


class SupervisedPlotGenerator(PlotGenerator):
	def __init__(self,
		spec,
		n_trials,
		data_fracs,
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

		:param data_fracs: Proportions of the overall size
			of the dataset to use
			(the horizontal axis on the three plots).
		:type data_fracs: List(float)

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

		:param perf_eval_kwargs: Extra keyword arguments to pass to
			perf_eval_fn
		:type perf_eval_kwargs: dict

		:param constraint_eval_kwargs: Extra keyword arguments to pass to
			the constraint_eval_fns
		:type constraint_eval_kwargs: dict
		"""

		super().__init__(spec=spec,
			n_trials=n_trials,
			data_fracs=data_fracs,
			datagen_method=datagen_method,
			perf_eval_fn=perf_eval_fn,
			results_dir=results_dir,
			n_workers=n_workers,
			constraint_eval_fns=constraint_eval_fns,
			perf_eval_kwargs=perf_eval_kwargs,
			constraint_eval_kwargs=constraint_eval_kwargs,
			)
		self.regime = 'supervised_learning'

	def run_seldonian_experiment(self,verbose=False):
		""" Run a supervised Seldonian experiment using the spec attribute
		assigned to the class in __init__().

		:param verbose: Whether to display results to stdout 
			while the Seldonian algorithms are running in each trial
		:type verbose: bool, defaults to False
		"""

		dataset = self.spec.dataset

		label_column = dataset.label_column
		sensitive_column_names = dataset.sensitive_column_names
		include_sensitive_columns = dataset.include_sensitive_columns
		include_intercept_term = dataset.include_intercept_term
		
		if self.datagen_method == 'resample':
			# Generate n_trials resampled datasets of full length
			# These will be cropped to data_frac fractional size
			print("generating resampled datasets")
			generate_resampled_datasets(dataset.df,
				self.n_trials,
				self.results_dir,
				file_format='pkl')
			print("Done generating resampled datasets")
			print()

		run_seldonian_kwargs = dict(
			spec=self.spec,
			data_fracs=self.data_fracs,
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
		sd_exp = SeldonianExperiment(model_name='qsa',
			results_dir=self.results_dir)

		sd_exp.run_experiment(**run_seldonian_kwargs)
		return

	def run_baseline_experiment(self,model_name,verbose=False):
		""" Run a supervised Seldonian experiment using the spec attribute
		assigned to the class in __init__().

		:param model_name: The name of the baseline model to use
	
		:type model_name: str

		:param verbose: Whether to display results to stdout 
			while the Seldonian algorithms are running in each trial
		:type verbose: bool, defaults to False
		"""

		dataset = self.spec.dataset

		label_column = dataset.label_column
		sensitive_column_names = dataset.sensitive_column_names
		include_sensitive_columns = dataset.include_sensitive_columns
		include_intercept_term = dataset.include_intercept_term
		
		if self.datagen_method == 'resample':
			# Generate n_trials resampled datasets of full length
			# These will be cropped to data_frac fractional size
			print("checking for resampled datasets")
			generate_resampled_datasets(dataset.df,
				self.n_trials,
				self.results_dir,
				file_format='pkl')
			print("Done checking for resampled datasets")
			print()

		run_baseline_kwargs = dict(
			spec=self.spec,
			data_fracs=self.data_fracs,
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
		bl_exp = BaselineExperiment(model_name=model_name,
			results_dir=self.results_dir)

		bl_exp.run_experiment(**run_baseline_kwargs)
		return

	def run_fairlearn_experiment(self,
		fairlearn_sensitive_feature_names,
		fairlearn_constraint_name,
		fairlearn_epsilon_constraint,
		fairlearn_epsilon_eval,
		fairlearn_eval_kwargs={},
		verbose=False):
		""" Run a supervised experiment using the fairlearn
		library 

		:param verbose: Whether to display results to stdout 
			while the fairlearn algorithms are running in each trial
		:type verbose: bool, defaults to False
		"""

		dataset = self.spec.dataset

		label_column = dataset.label_column
		sensitive_column_names = dataset.sensitive_column_names
		include_sensitive_columns = dataset.include_sensitive_columns
		include_intercept_term = dataset.include_intercept_term
		
		if self.datagen_method == 'resample':
			# Generate n_trials resampled datasets of full length
			# These will be cropped to data_frac fractional size
			print("Checking for resampled datasets")
			generate_resampled_datasets(dataset.df,
				self.n_trials,
				self.results_dir,
				file_format='pkl')
			print("Done generating resampled datasets")
			print()

		run_fairlearn_kwargs = dict(
			spec=self.spec,
			data_fracs=self.data_fracs,
			n_trials=self.n_trials,
			n_workers=self.n_workers,
			datagen_method=self.datagen_method,
			fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
			fairlearn_constraint_name=fairlearn_constraint_name,
			fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
			fairlearn_epsilon_eval=fairlearn_epsilon_eval,
			fairlearn_eval_kwargs=fairlearn_eval_kwargs,
			perf_eval_fn=self.perf_eval_fn,
			perf_eval_kwargs=self.perf_eval_kwargs,
			constraint_eval_fns=self.constraint_eval_fns,
			constraint_eval_kwargs=self.constraint_eval_kwargs,
			verbose=verbose,
			)

		## Run experiment 
		fl_exp = FairlearnExperiment(
			results_dir=self.results_dir,
			fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
			)

		fl_exp.run_experiment(**run_fairlearn_kwargs)
		return


class RLPlotGenerator(PlotGenerator):
	def __init__(self,
		spec,
		n_trials,
		data_fracs,
		datagen_method,
		hyperparameter_and_setting_dict,
		perf_eval_fn,
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

		:param data_fracs: Proportions of the overall size
			of the dataset to use
			(the horizontal axis on the three plots).
		:type data_fracs: List(float)

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

		:param perf_eval_kwargs: Extra keyword arguments to pass to
			perf_eval_fn
		:type perf_eval_kwargs: dict

		:param constraint_eval_kwargs: Extra keyword arguments to pass to
			the constraint_eval_fns
		:type constraint_eval_kwargs: dict
		"""

		super().__init__(spec=spec,
			n_trials=n_trials,
			data_fracs=data_fracs,
			datagen_method=datagen_method,
			perf_eval_fn=perf_eval_fn,
			results_dir=results_dir,
			n_workers=n_workers,
			constraint_eval_fns=constraint_eval_fns,
			perf_eval_kwargs=perf_eval_kwargs,
			constraint_eval_kwargs=constraint_eval_kwargs,
			)
		
		self.regime = 'reinforcement_learning'
		self.hyperparameter_and_setting_dict=hyperparameter_and_setting_dict

	def run_seldonian_experiment(self,verbose=False):
		""" Run an RL Seldonian experiment using the spec attribute
		assigned to the class in __init__().

		:param verbose: Whether to display results to stdout 
			while the Seldonian algorithms are running in each trial
		:type verbose: bool, defaults to False
		"""
		from seldonian.RL.RL_runner import run_trial
		print("Running experiment")
		dataset = self.spec.dataset
		
		if self.datagen_method == 'generate_episodes':
			# generate full-size datasets for each trial so that 
			# we can reference them for each data_frac
			save_dir = os.path.join(self.results_dir,'regenerated_datasets')
			os.makedirs(save_dir,exist_ok=True)
			print("generating new episodes for each trial")
			for trial_i in range(self.n_trials):
				print(f"Trial: {trial_i}")
				savename = os.path.join(save_dir,f'regenerated_data_trial{trial_i}.pkl')
				if not os.path.exists(savename):
					episodes,agent = run_trial(
						self.hyperparameter_and_setting_dict,
						parallel=False)
					# Save episodes
					save_pickle(savename,episodes,verbose=True)
				else:
					print(f"{savename} already created")

		run_seldonian_kwargs = dict(
			spec=self.spec,
			data_fracs=self.data_fracs,
			n_trials=self.n_trials,
			n_workers=self.n_workers,
			datagen_method=self.datagen_method,
			hyperparameter_and_setting_dict=self.hyperparameter_and_setting_dict,
			constraint_eval_fns=self.constraint_eval_fns,
			constraint_eval_kwargs=self.constraint_eval_kwargs,
			perf_eval_fn=self.perf_eval_fn,
			perf_eval_kwargs=self.perf_eval_kwargs,
			verbose=verbose,
			)


		# # ## Run experiment 
		sd_exp = SeldonianExperiment(model_name='qsa',
			results_dir=self.results_dir)

		sd_exp.run_experiment(**run_seldonian_kwargs)

		



	
	
