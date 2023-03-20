## Run headless mnist experiments (not yet documented on the website)

### Epsilon is the accuracy threshold:
# 

import argparse
import numpy as np
import os
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.base_example import BaseExample
from experiments.utils import multiclass_accuracy, multiclass_logistic_loss
from seldonian.utils.io_utils import load_pickle


def headless_mnist_example(
    spec_rootdir,
    results_base_dir,
    accuracy_thresholds=[0.95],
    n_trials=50,
    data_fracs=np.logspace(-3,0,15),
    baselines = ["random_classifier"],
    performance_metric="log_loss",
    n_workers=1,
):  
    if performance_metric == "accuracy":
        perf_eval_fn = multiclass_accuracy
    elif performance_metric == 'log_loss':
        perf_eval_fn = multiclass_logistic_loss
    else:
        raise NotImplementedError(
            f"Performance metric {performance_metric} not supported for this example")

    for accuracy_threshold in accuracy_thresholds:
        specfile = os.path.join(
            spec_rootdir,
            f"headless_mnist_accuracy_{accuracy_threshold}.pkl"
        )
        spec = load_pickle(specfile)
        results_dir = os.path.join(results_base_dir,
            f"headless_mnist_{constraint}_{epsilon}_{performance_metric}")
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
    parser.add_argument('--accuracy_threshold', help='Accuracy >= accuracy_threshold', required=True)
    parser.add_argument('--n_trials', help='Number of trials to run', required=True)
    parser.add_argument('--n_workers', help='Number of workers to use', required=True)
    parser.add_argument('--include_baselines', help='verbose', action="store_true")
    parser.add_argument('--verbose', help='verbose', action="store_true")

    args = parser.parse_args()

    accuracy_threshold = float(args.accuracy_threshold)
    n_trials = int(args.n_trials)
    n_workers = int(args.n_workers)
    include_baselines = args.include_baselines
    verbose = args.verbose

    if include_baselines:
        baselines = ["random_classifier","logistic_regression"]
    else:
        baselines = []

    performance_metric="accuracy"

    results_base_dir = f"./results"

    headless_mnist_example(
        spec_rootdir="./data/spec",
        results_base_dir=results_base_dir,
        accuracy_thresholds=[accuracy_threshold],
        n_trials=n_trials,
        performance_metric=performance_metric
    )
    
