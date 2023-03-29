## This script runs the following examples with different amounts of data splits.

# Comment out examples you don't want to run

import os
import time 
import numpy as np
from datetime import timedelta
from lie_detection.generate_experiment_plots import ds_lie_detection_example
from loans.generate_experiment_plots import ds_loans_example
from gpa_science_classification.generate_experiment_plots import ds_gpa_example

### MODIFY HOW TO RUN THE EXAMPLES ###

## Comment out any examples you don't want to run
examples_to_run = [
	# "loans",
	# "gpa_science_classification",
	"lie_detection",
]

## The path on your machine that will serve as the 
## parent directory containing all of the experiment results
results_root_dir = "./example_results"

## Details for running each example.
## Modify as needed
example_setup_dict = {
	'loans':{
		'spec_rootdir':'loans/data/spec',
		'n_trials':50, # Number of trials per data fraction
		'n_workers':30, # Number of CPUs to use 
        'constraints':[
            "disparate_impact",
            "disparate_impact_fairlearndef",
            "equalized_odds",
            ],
        'all_frac_data_in_safety':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
	},
	'lie_detection':{
		'spec_rootdir':'lie_detection/data/spec', 
		'n_trials':10, # Number of trials per data fraction
		'n_workers':30, # Number of CPUs to use 
        'constraints':[
            "disparate_impact",
            "predictive_equality",
            "equal_opportunity",
            "overall_accuracy_equality",
            ], 
        'epsilons':[ # the actual values are 1-this in the constraints
            0.2,
            0.1,
            0.05
            ], 
        'all_frac_data_in_safety': [0.2, 0.4, 0.6, 0.8]
	},
	'gpa_science_classification':{
		'spec_rootdir':'gpa_science_classification/data/spec',
		'n_trials':10, # Number of trials per data fraction
		'n_workers':30, # Number of CPUs to use 
        'constraints':[
            "disparate_impact",
            "demographic_parity",
            "equalized_odds",
            "equal_opportunity",
            "predictive_equality"
            ],
        'all_frac_data_in_safety': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
	},
}

### DO NOT MODIFY BELOW ###

if __name__ == "__main__":
	start_time = time.time()
	for example in examples_to_run:
		spec_rootdir = example_setup_dict[example]['spec_rootdir']
		results_example_basedir = os.path.join(
				results_root_dir,example)

		if example == 'loans':
			ds_loans_example(
			    spec_rootdir=spec_rootdir,
			    results_base_dir=results_example_basedir,
			    constraints=example_setup_dict[example]['constraints'],
			    n_trials=example_setup_dict[example]['n_trials'],
			    data_fracs=np.logspace(-3, 0, 15),
			    baselines=[],
			    include_fairlearn_models=False,
			    performance_metric="log_loss",
			    n_workers=example_setup_dict[example]['n_trials'],
			    all_frac_data_in_safety=example_setup_dict[example]['all_frac_data_in_safety'],
			)

		if example == 'lie_detection':	
			ds_lie_detection_example(
				spec_rootdir=spec_rootdir,
				results_base_dir=results_example_basedir,
			    constraints=example_setup_dict[example]['constraints'],
			    n_trials=example_setup_dict[example]['n_trials'],
			    data_fracs=np.logspace(-3,0,15),
			    baselines=[],
			    epsilons=example_setup_dict[example]['epsilons'],
			    performance_metric="accuracy",
			    n_workers=example_setup_dict[example]['n_trials'],
			    all_frac_data_in_safety=example_setup_dict[example]['all_frac_data_in_safety'],
			)

		if example == 'gpa_science_classification':
			ds_gpa_example(
			    spec_rootdir=spec_rootdir,
			    results_base_dir=results_example_basedir,
			    constraints=example_setup_dict[example]['constraints'],
			    n_trials=example_setup_dict[example]['n_trials'],
			    data_fracs=np.logspace(-4,0,15),
			    baselines=[],
			    include_fairlearn_models=False,
			    performance_metric="accuracy",
			    n_workers=example_setup_dict[example]['n_trials'],
			    all_frac_data_in_safety=example_setup_dict[example]['all_frac_data_in_safety'],
			)

		end_time = time.time()
		elapsed = end_time - start_time
		print("Time elapsed: " + str(timedelta(seconds=elapsed)))
