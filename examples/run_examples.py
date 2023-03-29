## This script runs one or more of the examples with a single command line call

# Comment out examples you don't want to run

import os
import numpy as np
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
results_root_dir = "./example_results"

## Details for running each example.
## Modify as needed
example_setup_dict = {
	'loans':{
		'spec_rootdir':'loans/data/spec',
		'n_trials':50, # Number of trials per data fraction
		'n_workers':8, # Number of CPUs to use 
	},
	'lie_detection':{
		'spec_rootdir':'lie_detection/data/spec', 
		'n_trials':50, # Number of trials per data fraction
		'n_workers':8, # Number of CPUs to use 

	},
	'gpa_science_classification':{
		'spec_rootdir':'gpa_science_classification/data/spec',
		'n_trials':50, # Number of trials per data fraction
		'n_workers':8, # Number of CPUs to use 
	},

}

### DO NOT MODIFY BELOW ###

if __name__ == "__main__":
	for example in examples_to_run:
		spec_rootdir = example_setup_dict[example]['spec_rootdir']
		results_example_basedir = os.path.join(
				results_root_dir,example)
		
		if example == 'lie_detection':	
			lie_detection_example(
				spec_rootdir=spec_rootdir,
				results_base_dir=results_example_basedir,
			    constraints = [
			        "disparate_impact",
			        "predictive_equality",
			        "equal_opportunity",
			        "overall_accuracy_equality",
			    ],
			    n_trials=example_setup_dict[example]['n_trials'],
			    data_fracs=np.logspace(-3,0,15),
			    baselines = ["random_classifier","logistic_regression"],
			    epsilons=[0.2,0.1,0.05], # the actual values are 1-this in the constraints
			    performance_metric="accuracy",
			    n_workers=example_setup_dict[example]['n_trials'],
			)

		if example == 'loans':
			loans_example(
			    spec_rootdir=spec_rootdir,
			    results_base_dir=results_example_basedir,
			    constraints=[
			        "disparate_impact",
			        "disparate_impact_fairlearndef",
			        "equalized_odds",
			    ],
			    n_trials=example_setup_dict[example]['n_trials'],
			    data_fracs=np.logspace(-3, 0, 15),
			    baselines=["random_classifier", "logistic_regression"],
			    include_fairlearn_models=True,
			    performance_metric="log_loss",
			    n_workers=example_setup_dict[example]['n_trials'],
			)

		if example == 'gpa_science_classification':
			gpa_example(
			    spec_rootdir=spec_rootdir,
			    results_base_dir=results_example_basedir,
			    constraints=[
			        "disparate_impact",
			        "demographic_parity",
			        "equalized_odds",
			        "equal_opportunity",
			        "predictive_equality"
			    ],
			    n_trials=example_setup_dict[example]['n_trials'],
			    data_fracs=np.logspace(-4,0,15),
			    baselines=["random_classifier", "logistic_regression"],
			    include_fairlearn_models=True,
			    performance_metric="accuracy",
			    n_workers=example_setup_dict[example]['n_trials'],
			)
