import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
	# Parameter setup
	run_experiments = False
	make_plots = True
	save_plot = True

	performance_metric = 'Mean Squared Error'
	# n_trials = 20
	n_trials = 25
	data_fracs = np.logspace(-3,0,10)
	n_workers = 7
	verbose=True
	results_dir = f'../../results/housing_prices_2022Oct19_eps0.2'
	plot_savename = os.path.join(results_dir,f'housing_prices_3plots.png')

	# Load spec
	specfile = f'../../../engine-repo-dev/examples/housing_prices/spec.pkl'
	spec = load_pickle(specfile)

	os.makedirs(results_dir,exist_ok=True)

	# Use entire original dataset as ground truth for test set
	dataset = spec.dataset
	label_column = dataset.label_column
	include_sensitive_columns = dataset.include_sensitive_columns

	test_features = dataset.df.loc[:,
		dataset.df.columns != label_column]
	test_labels = dataset.df[label_column]

	if not include_sensitive_columns:
		test_features = test_features.drop(
			columns=dataset.sensitive_column_names) 

	# Setup performance evaluation function and kwargs 
	# of the performance evaluation function

	def perf_eval_fn(y_pred,y,**kwargs):
		return mean_squared_error(y,y_pred)
		
	perf_eval_kwargs = {
		'X':test_features,
		'y':test_labels,
		}

	plot_generator = SupervisedPlotGenerator(
		spec=spec,
		n_trials=n_trials,
		data_fracs=data_fracs,
		n_workers=n_workers,
		datagen_method='resample',
		perf_eval_fn=perf_eval_fn,
		constraint_eval_fns=[],
		results_dir=results_dir,
		perf_eval_kwargs=perf_eval_kwargs,
		)

	if run_experiments:
		# Baseline models
		plot_generator.run_baseline_experiment(
			model_name='linear_regression',verbose=True)

		# Seldonian experiment
		plot_generator.run_seldonian_experiment(verbose=verbose)


	if make_plots:
		if save_plot:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric,
				savename=plot_savename)
		else:
			plot_generator.make_plots(fontsize=12,legend_fontsize=8,
				performance_label=performance_metric)