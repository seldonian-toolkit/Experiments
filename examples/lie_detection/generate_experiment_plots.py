## Run lie detection experiments as described in this example: 
## https://seldonian.cs.umass.edu/Tutorials/examples/lie_detection/
### Possible constraint names: 
# [
#     "disparate_impact",
#     "predictive_equality",
#     "equal_opportunity",
#     "overall_accuracy_equality",
# ]
### Possible epsilon values:
# 0.2, 0.1, 0.05 

import argparse
import numpy as np
import os
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.base_example import BaseExample
from experiments.perf_eval_funcs import probabilistic_accuracy
from experiments.baselines.logistic_regression import BinaryLogisticRegressionBaseline
from experiments.baselines.random_forest import RandomForestClassifierBaseline
from experiments.baselines.random_classifiers import (
    UniformRandomClassifierBaseline)
from seldonian.utils.io_utils import load_pickle


def lie_detection_example(
    spec_rootdir,
    results_base_dir,
    constraints = [
        "disparate_impact",
        "predictive_equality",
        "equal_opportunity",
        "overall_accuracy_equality",
    ],
    epsilons=[0.2,0.1,0.05],
    n_trials=50,
    data_fracs=np.logspace(-3,0,15),
    baselines = [UniformRandomClassifierBaseline(),BinaryLogisticRegressionBaseline()],
    performance_metric="accuracy",
    n_workers=1,
):  
    if performance_metric == "accuracy":
        perf_eval_fn = probabilistic_accuracy
    else:
        raise NotImplementedError(
            "Performance metric must be 'accuracy' for this example")

    for constraint in constraints:
        for epsilon in epsilons:
            specfile = os.path.join(
                spec_rootdir,
                f"lie_detection_{constraint}_{epsilon}.pkl"
            )
            spec = load_pickle(specfile)
            results_dir = os.path.join(results_base_dir,
                f"lie_detection_{constraint}_{epsilon}_{performance_metric}")
            plot_savename = os.path.join(
                results_dir, f"{constraint}_{epsilon}_{performance_metric}.pdf"
            )

            ex = BaseExample(spec=spec)

            ex.run(
                n_trials=n_trials,
                data_fracs=data_fracs,
                results_dir=results_dir,
                perf_eval_fn=perf_eval_fn,
                n_workers=n_workers,
                datagen_method="resample",
                verbose=False,
                baselines=baselines,
                performance_label=performance_metric,
                performance_yscale="linear",
                plot_savename=plot_savename,
                plot_fontsize=12,
                legend_fontsize=8,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--constraint', help='Constraint to run', required=True)
    parser.add_argument('--epsilon', help='Constraint to run', required=True)
    parser.add_argument('--n_trials', help='Number of trials to run', required=True)
    parser.add_argument('--n_workers', help='Number of workers to use', required=True)
    parser.add_argument('--include_baselines', help='verbose', action="store_true")
    parser.add_argument('--verbose', help='verbose', action="store_true")

    args = parser.parse_args()

    constraint = args.constraint
    epsilon = float(args.epsilon)
    n_trials = int(args.n_trials)
    n_workers = int(args.n_workers)
    include_baselines = args.include_baselines
    verbose = args.verbose

    if include_baselines:
        baselines = [
            UniformRandomClassifierBaseline(),
            BinaryLogisticRegressionBaseline(),
            RandomForestClassifierBaseline()
        ]
    else:
        baselines = []

    performance_metric="accuracy"

    results_base_dir = f"./results"

    lie_detection_example(
        spec_rootdir="./data/spec",
        results_base_dir=results_base_dir,
        constraints = [constraint],
        epsilons=[epsilon],
        n_trials=n_trials,
        baselines=baselines,
        performance_metric=performance_metric,
        n_workers=n_workers
    )
    
