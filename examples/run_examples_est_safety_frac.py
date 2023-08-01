## This script runs one or more of the examples with a single command line call

# Comment out examples you don't want to run

import os
import time
import numpy as np
from datetime import timedelta
from seldonian.spec import HyperparameterSelectionSpec
from lie_detection.generate_experiment_plots import lie_detection_example
from loans.generate_experiment_plots import loans_example
from gpa_science_classification.generate_experiment_plots import gpa_example

### MODIFY HOW TO RUN THE EXAMPLES ###

## Comment out any examples you don't want to run
examples_to_run = [
	"loans",
	# "gpa_science_classification",
	# "lie_detection",
]

## The path on your machine that will serve as the 
## parent directory containing all of the experiment results
results_root_dir = "./results/example_select_results"
os.makedirs(results_root_dir, exist_ok=True)

## Set some shared hyperparamers.
N_TRIALS = 5
N_WORKERS = 1
N_BOOTSTRAP_TRIALS = 50
N_BOOTSTRAP_WORKERS = 60
ALL_FRAC_DATA_IN_SAFETY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

## Create HyperparameterSelectionSpec.
hyperparam_select_spec = HyperparameterSelectionSpec(
        n_bootstrap_trials=N_BOOTSTRAP_TRIALS,
        all_frac_data_in_safety=ALL_FRAC_DATA_IN_SAFETY,
        n_bootstrap_workers=N_BOOTSTRAP_WORKERS
)

## Details for running each example.
## Modify as needed
example_setup_dict = {
	'loans':{
		'spec_rootdir':'loans/data/spec',
		'n_trials':N_TRIALS, # Number of trials per data fraction
		'n_workers':N_WORKERS, # Number of CPUs to use 
		'constraints':[
		    "disparate_impact",
		    # "disparate_impact_fairlearndef",
		    # "equalized_odds",
		],
		'hyperparam_select_spec': hyperparam_select_spec,
	},
	'lie_detection':{
		'spec_rootdir':'lie_detection/data/spec', 
		'n_trials':N_TRIALS, # Number of trials per data fraction
		'n_workers':N_WORKERS, # Number of CPUs to use 
                'constraints':[
                    "disparate_impact",
                    "predictive_equality",
                    "equal_opportunity",
                    "overall_accuracy_equality",
                    ],
                'epsilons':[  # the actual values are 1-this in the constraints
                    0.2,
                    0.1,
                    0.05,
                    ],
		'hyperparam_select_spec': hyperparam_select_spec,

	},
	'gpa_science_classification':{
		'spec_rootdir':'gpa_science_classification/data/spec',
		'n_trials':N_TRIALS, # Number of trials per data fraction
		'n_workers':N_WORKERS, # Number of CPUs to use 
		'constraints':[
		    "disparate_impact",
		    # "demographic_parity",
		    # "equalized_odds",
		    # "equal_opportunity",
		    # "predictive_equality"
		],
		'hyperparam_select_spec': hyperparam_select_spec,
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
                    loans_example(
                        spec_rootdir=spec_rootdir,
                        results_base_dir=results_example_basedir,
                        constraints=example_setup_dict[example]['constraints'],
                        n_trials=example_setup_dict[example]['n_trials'],
                        data_fracs=np.logspace(-3, 0, 15)[5:], 
                        baselines=[],
                        include_fairlearn_models=False,
                        performance_metric="log_loss",
                        n_workers=example_setup_dict[example]['n_workers'],
                        hyperparam_select_spec=example_setup_dict[example]['hyperparam_select_spec'],
                    )

            if example == 'lie_detection':	
                    lie_detection_example(
                            spec_rootdggir=spec_rootdir,
                            results_base_dir=results_example_basedir,
                        constraints = example_setup_dict[example]['constraints'],
                        n_trials=example_setup_dict[example]['n_trials'],
                        data_fracs=np.logspace(-3,0,15),
                        baselines = [],
                        epsilons=example_setup_dict[example]['epsilons'],
                        performance_metric="accuracy",
                        n_workers=example_setup_dict[example]['n_workers'],
                        hyperparam_select_spec=example_setup_dict[example]['hyperparam_select_spec'],
                    )

            if example == 'gpa_science_classification':
                    gpa_example(
                        spec_rootdir=spec_rootdir,
                        results_base_dir=results_example_basedir,
                        constraints=example_setup_dict[example]['constraints'],
                        n_trials=example_setup_dict[example]['n_trials'],
                        data_fracs=np.logspace(-4,0,15),
                        baselines=[],
                        include_fairlearn_models=False,
                        performance_metric="accuracy",
                        n_workers=example_setup_dict[example]['n_workers'],
                        hyperparam_select_spec=example_setup_dict[example]['hyperparam_select_spec'],
                    )

    end_time = time.time()
    elapsed = end_time - start_time
    print("Time elapsed: " + str(timedelta(seconds=elapsed)))
