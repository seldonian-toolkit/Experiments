""" Module containing base class for running examples """
import os
from .generate_plots import SupervisedPlotGenerator


class BaseExample:
    def __init__(self, spec):
        """Base class for running experiments

        :param spec: specification object created using the Seldonian Engine.
        """
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
        baselines=[],
        model_label_dict={},
        include_fairlearn_models=False,
        fairlearn_kwargs={},
        performance_label="performance",
        performance_yscale="linear",
        plot_savename=None,
        plot_save_format="pdf",
        include_legend=True,
        plot_fontsize=12,
        legend_fontsize=8,
        verbose=False,
    ):
        """Run the experiments for this example.
        Runs any baseline models included in baselines
        parameter first. Then produces the three plots.

        :param n_trials: The number of trials for the experiments
        :param data_fracs: The data fractions for the experiments
        :param results_dir: Directory for saving results files
        :param perf_eval_fn: Performance evaluation function
        :param n_workers: Number of parallel processors to use 
        :param datagen_method: Method for generating the trial data
        :param baselines: List of baseline models to include
        :param model_label_dict: Dictionary mapping model names (see model.model_name)
            to display name in the 3 plots legend.
        :param include_fairlearn_models: Whether to include fairlearn baseline models
        :type include_fairlearn_models: Bool
        :param performance_label: Label to use on the performance plot (left-most plot)
        :type performance_label: str
        :param performance_yscale: How to scale the y-axis on the performance plot. 
            Options are "linear" and "log"
        :type performance_yscale: str
        :param plot_savename: If provided, the filepath where the three plots will be saved
        :param plot_save_format: "pdf" or "png"
        :param include_legend: Whether to include legend on the 3 plots
        :type include_legend: bool
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
            fairlearn_constraint_name = fairlearn_kwargs["fairlearn_constraint_name"]
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
