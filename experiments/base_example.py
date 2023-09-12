""" Module containing base class for running examples """
import os
from .generate_plots import SupervisedPlotGenerator


class BaseExample:
    def __init__(self, spec):
        """Base class for running experiments"""
        self.spec = spec
        self.regime = self.spec.dataset.regime

    def run(
        self,
        n_trials,
        data_fracs,
        results_dir,
        perf_eval_fn,
        n_workers=1,
        datagen_method="resample",
        verbose=False,
        baselines=[],
        model_label_dict = {},
        include_fairlearn_models=False,
        fairlearn_kwargs={},
        performance_label="performance",
        performance_yscale="linear",
        plot_savename=None,
        plot_save_format="pdf",
        include_legend=True,
        plot_fontsize=12,
        legend_fontsize=8,
    ):
        """Run the experiment for this example.
        Runs any baseline models included in baselines
        parameter first. Then produces the three plots.
        """
        os.makedirs(results_dir, exist_ok=True)

        dataset = self.spec.dataset
        test_features = dataset.features
        test_labels = dataset.labels

        perf_eval_kwargs = {
            "X": test_features,
            "y": test_labels,
        }
        if self.regime == "supervised_learning":
            plot_generator = SupervisedPlotGenerator(
                spec=self.spec,
                n_trials=n_trials,
                data_fracs=data_fracs,
                n_workers=n_workers,
                datagen_method=datagen_method,
                perf_eval_fn=perf_eval_fn,
                constraint_eval_fns=[],
                results_dir=results_dir,
                perf_eval_kwargs=perf_eval_kwargs,
            )

        # Baselines first
        for baseline_model in baselines:
            plot_generator.run_baseline_experiment(
                baseline_model=baseline_model, verbose=verbose
            )
        # Check if fairlearn requested
        if include_fairlearn_models:
            fairlearn_sensitive_feature_names = fairlearn_kwargs[
                "fairlearn_sensitive_feature_names"
            ]
            fairlearn_constraint_name = fairlearn_kwargs[
                "fairlearn_constraint_name"
            ]
            fairlearn_constraint_epsilons = fairlearn_kwargs[
                "fairlearn_constraint_epsilons"
            ]
            fairlearn_epsilon_eval = fairlearn_kwargs["fairlearn_epsilon_eval"]
            fairlearn_eval_method = fairlearn_kwargs["fairlearn_eval_method"]

            fairlearn_sensitive_col_indices = [
                dataset.sensitive_col_names.index(col)
                for col in fairlearn_sensitive_feature_names
            ]
            fairlearn_sensitive_features = dataset.sensitive_attrs[
                :, fairlearn_sensitive_col_indices
            ]
            # Setup ground truth test dataset for Fairlearn
            test_features_fairlearn = test_features
            fairlearn_eval_kwargs = {
                "X": test_features_fairlearn,
                "y": test_labels,
                "sensitive_features": fairlearn_sensitive_features,
                "eval_method": fairlearn_eval_method,
            }
            for fairlearn_epsilon_constraint in fairlearn_constraint_epsilons:
                plot_generator.run_fairlearn_experiment(
                    verbose=verbose,
                    fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
                    fairlearn_constraint_name=fairlearn_constraint_name,
                    fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
                    fairlearn_epsilon_eval=fairlearn_epsilon_eval,
                    fairlearn_eval_kwargs=fairlearn_eval_kwargs,
                )
        # Run Seldonian experiment
        plot_generator.run_seldonian_experiment(verbose=verbose)

        plot_generator.make_plots(
            model_label_dict=model_label_dict,
            fontsize=plot_fontsize,
            include_legend=include_legend,
            legend_fontsize=legend_fontsize,
            performance_label=performance_label,
            performance_yscale=performance_yscale,
            save_format=plot_save_format,
            savename=plot_savename,
        )