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
import numpy as np
import os
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.base_example import BaseExample
from experiments.utils import deterministic_accuracy
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss

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
    baselines=["random_classifier", "logistic_regression"],
    include_fairlearn_models=True,
    performance_metric="accuracy",
    n_workers=1,
    hyperparam_select_spec=None,
):
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
            results_dir, f"{constraint}_{epsilon}_{performance_metric}.pdf"
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
            hyperparam_select_spec=hyperparam_select_spec,
            verbose=False,
            baselines=baselines,
            include_fairlearn_models=include_fairlearn_models,
            fairlearn_kwargs=fairlearn_kwargs,
            performance_label=performance_metric,
            performance_yscale="log",
            plot_savename=plot_savename,
            plot_fontsize=12,
            legend_fontsize=8,
        )


# Run the gpa example with different fractions of data in safety
def gpa_example_all_safety_frac(
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
    baselines=["random_classifier", "logistic_regression"],
    include_fairlearn_models=True,
    performance_metric="accuracy",
    n_workers=1,
    all_frac_data_in_safety=[0.6],
    make_plot=False
):
    if performance_metric == "accuracy":
        perf_eval_fn = deterministic_accuracy
    else:
        raise NotImplementedError(
            "Performance metric must be 'accuracy' for this example")

    for frac_data_in_safety in all_frac_data_in_safety:
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

            # Modify the fraction of safety data.
            spec.frac_data_in_safety = frac_data_in_safety

            # Change results dir to include safety data.
            results_dir = os.path.join(results_base_dir,
                    f"gpa_science_classification_{constraint}_{epsilon}_{performance_metric}/safety%d" % (frac_data_in_safety * 100))
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
                include_fairlearn_models=include_fairlearn_models,
                fairlearn_kwargs=fairlearn_kwargs,
                performance_label=performance_metric,
                performance_yscale="log",
                make_plot=make_plot,
                plot_savename=plot_savename,
                plot_fontsize=12,
                legend_fontsize=8,
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
    parser.add_argument("--verbose", help="verbose", action="store_true")

    args = parser.parse_args()

    constraint = args.constraint
    n_trials = int(args.n_trials)
    n_workers = int(args.n_workers)
    include_baselines = args.include_baselines
    include_fairlearn_models = args.include_fairlearn_models
    verbose = args.verbose

    if include_baselines:
        baselines = ["random_classifier", "logistic_regression"]
    else:
        baselines = []

    results_base_dir = f"./results"

    gpa_example(
        spec_rootdir="data/spec",
        results_dir=results_dir,
        constraints=[constraint],
        n_trials=n_trials,
        data_fracs=np.logspace(-3, 0, 15),
        baselines=baselines,
        include_fairlearn_models=include_fairlearn_models,
        performance_metric="accuracy",
        n_workers=n_workers,
    )
