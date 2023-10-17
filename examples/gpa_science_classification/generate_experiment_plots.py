## Run GPA experiments as described in this tutorial:
# https://seldonian.cs.umass.edu/Tutorials/tutorials/science_GPA_tutorial/
### Possible constraint names:
# [
#     "disparate_impact",
#     "demographic_parity",
#     "equalized_odds",
#     "equal_opportunity",
#     "predictive_equality"
# ]

import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss

from experiments.generate_plots import SupervisedPlotGenerator
from experiments.base_example import BaseExample
from experiments.baselines.logistic_regression import BinaryLogisticRegressionBaseline
from experiments.baselines.random_classifiers import (
    UniformRandomClassifierBaseline)
from experiments.perf_eval_funcs import deterministic_accuracy



def gpa_example(
    spec_rootdir,
    results_base_dir,
    constraints=[
        "disparate_impact",
        "demographic_parity",
        "equalized_odds",
        "equal_opportunity",
        "predictive_equality"
    ],
    n_trials=50,
    data_fracs=np.logspace(-4,0,15),
    baselines=[UniformRandomClassifierBaseline(),BinaryLogisticRegressionBaseline()],
    include_fairlearn_models=True,
    performance_metric="accuracy",
    plot_save_format="pdf",
    include_legend=True,
    model_label_dict={},
    n_workers=1,
):
    def initial_solution_fn(model,X,Y):
        return model.fit(X,Y)
    if performance_metric == "accuracy":
        perf_eval_fn = deterministic_accuracy
    else:
        raise NotImplementedError(
            "Performance metric must be 'accuracy' for this example")

    for constraint in constraints:
        if constraint == "disparate_impact":
            epsilon = 0.8
        elif constraint in ["demographic_parity","equal_opportunity","predictive_equality"]:
            epsilon = 0.2
        elif constraint == "equalized_odds":
            epsilon = 0.35

        fairlearn_eval_method = "two-groups" # e.g., PR | [M] vs. PR | [F]
        fairlearn_constraint_name = constraint

        specfile = os.path.join(spec_rootdir, 
            f"gpa_science_classification_{constraint}_{epsilon}_spec.pkl")
        spec = load_pickle(specfile)

        results_dir = os.path.join(results_base_dir,
                f"gpa_science_classification_{constraint}_{epsilon}_{performance_metric}")
        plot_savename = os.path.join(
            results_dir, f"{constraint}_{epsilon}_{performance_metric}.{plot_save_format}"
        )

        if include_fairlearn_models:
            fairlearn_constraint_epsilons = [0.01,0.1,1.0]

            fairlearn_kwargs = {
                "fairlearn_constraint_name": fairlearn_constraint_name,
                "fairlearn_constraint_epsilons": fairlearn_constraint_epsilons,
                "fairlearn_epsilon_eval": epsilon,
                "fairlearn_sensitive_feature_names": ["M"],
                "fairlearn_eval_method": fairlearn_eval_method,
            }

        else:
            fairlearn_kwargs = {}

        ex = BaseExample(spec=spec)

        ex.run(
            n_trials=n_trials,
            data_fracs=data_fracs,
            results_dir=results_dir,
            perf_eval_fn=perf_eval_fn,
            n_workers=n_workers,
            datagen_method="resample",
            verbose=False,
            model_label_dict=model_label_dict,
            baselines=baselines,
            include_fairlearn_models=include_fairlearn_models,
            fairlearn_kwargs=fairlearn_kwargs,
            performance_label=performance_metric,
            performance_yscale="linear",
            plot_savename=plot_savename,
            plot_save_format=plot_save_format,
            plot_fontsize=12,
            include_legend=include_legend,
            legend_fontsize=12,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--constraint", help="Constraint to run", required=True)
    parser.add_argument("--n_trials", help="Number of trials to run", required=True)
    parser.add_argument("--n_workers", help="Number of workers to use", required=True)
    parser.add_argument(
        "--include_baselines", help="whether to run baselines`", action="store_true"
    )
    parser.add_argument(
        "--include_fairlearn_models",
        help="whether to run fairlearn models",
        action="store_true",
    )
    parser.add_argument(
        "--include_legend",
        help="whether to run fairlearn models",
        action="store_true",
    )
    parser.add_argument("--verbose", help="verbose", action="store_true")

    args = parser.parse_args()

    constraint = args.constraint
    n_trials = int(args.n_trials)
    n_workers = int(args.n_workers)
    include_baselines = args.include_baselines
    include_legend = args.include_legend
    include_fairlearn_models = args.include_fairlearn_models
    verbose = args.verbose

    model_label_dict = {
        "qsa": "Quasi-Seldonian algorithm",
        "logistic_regression": "Logistic regression (no constraint)",
        "uniform_random": "Uniform random classifier",
        "fairlearn_eps0.01": "Fairlearn (eps=0.01)",
        "fairlearn_eps0.10": "Fairlearn (eps=0.1)",
        "fairlearn_eps1.00": "Fairlearn (eps=1.0)",
    }

    if include_baselines:
        baselines = [UniformRandomClassifierBaseline(),BinaryLogisticRegressionBaseline()]
    else:
        baselines = []

    results_base_dir = f"./results"

    gpa_example(
        spec_rootdir="data/spec",
        results_base_dir=results_base_dir,
        constraints=[constraint],
        n_trials=n_trials,
        data_fracs=np.logspace(-4, 0, 15),
        model_label_dict=model_label_dict,
        baselines=baselines,
        include_fairlearn_models=include_fairlearn_models,
        performance_metric="accuracy",
        n_workers=n_workers,
        include_legend=include_legend,
        plot_save_format="png"
    )
