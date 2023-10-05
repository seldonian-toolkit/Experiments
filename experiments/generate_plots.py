""" Module for making the three plots """

import os
import glob
import pickle
import autograd.numpy as np  # Thinly-wrapped version of Numpy
import pandas as pd
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import scipy
import matplotlib.pyplot as plt
from matplotlib import style

from seldonian.utils.io_utils import (load_pickle,save_pickle)
from seldonian.utils.stats_utils import tinv

from .experiments import BaselineExperiment, SeldonianExperiment, FairlearnExperiment
from .experiment_utils import (
    generate_resampled_datasets, 
    generate_behavior_policy_episodes, 
    has_failed
)

seldonian_model_set = set(["qsa","headless_qsa", "sa"])
plot_colormap = matplotlib.cm.get_cmap("tab10")
marker_list = ["s", "p", "d", "*", "x", "h", "+"]

np.seterr(invalid='ignore')


class PlotGenerator:
    def __init__(
        self,
        spec,
        n_trials,
        data_fracs,
        datagen_method,
        perf_eval_fn,
        results_dir,
        n_workers,
        n_bootstrap_trials=None,
        hyperparam_select_spec=None,
        constraint_eval_fns=[],
        perf_eval_kwargs={},
        constraint_eval_kwargs={},
        batch_epoch_dict={},
    ):
        """Class for running Seldonian experiments
        and generating the three plots:
        1) Performance
        2) Solution rate
        3) Failure rate
        all plotted vs. amount of data used

        :param spec: Specification object for running the
                Seldonian algorithm
        :type spec: seldonian.spec.Spec object

        :param n_trials: The number of times the
                Seldonian algorithm is run for each data fraction.
                Used for generating error bars
        :type n_trials: int

        :param data_fracs: Proportions of the overall size
                of the dataset to use
                (the horizontal axis on the three plots).
        :type data_fracs: List(float)

        :param datagen_method: Method for generating data that is used
                to run the Seldonian algorithm for each trial
        :type datagen_method: str, e.g. "resample"

        :param perf_eval_fn: Function used to evaluate the performance
                of the model obtained in each trial, with signature:
                func(theta,**kwargs), where theta is the solution
                from candidate selection
        :type perf_eval_fn: function or class method

        :param results_dir: The directory in which to save the results
        :type results_dir: str

        :param n_workers: The number of workers to use if
                using multiprocessing
        :type n_workers: int

        :param hyperparam_select_spec: Specification object for doing hyperparameter
                selection
        :type hyperparam_select_spec: seldonian.spec.HyperparameterSelectionSpec object

        :param constraint_eval_fns: List of functions used to evaluate
                the constraints on ground truth. If an empty list is provided,
                the constraints are evaluated using the parse tree
        :type constraint_eval_fns: List(function or class method),
                defaults to []

        :param perf_eval_kwargs: Extra keyword arguments to pass to
                perf_eval_fn
        :type perf_eval_kwargs: dict

        :param constraint_eval_kwargs: Extra keyword arguments to pass to
                the constraint_eval_fns
        :type constraint_eval_kwargs: dict

        :param batch_epoch_dict: Instruct batch sizes and n_epochs
                for each data frac
        :type batch_epoch_dict: dict
        """
        self.spec = spec
        self.n_trials = n_trials
        self.data_fracs = data_fracs
        self.datagen_method = datagen_method
        self.perf_eval_fn = perf_eval_fn
        self.results_dir = results_dir
        self.n_workers = n_workers
        self.n_bootstrap_trials = n_bootstrap_trials
        self.hyperparam_select_spec = hyperparam_select_spec
        self.constraint_eval_fns = constraint_eval_fns
        self.perf_eval_kwargs = perf_eval_kwargs
        self.constraint_eval_kwargs = constraint_eval_kwargs
        self.batch_epoch_dict = batch_epoch_dict

    def make_plots(
        self,
        model_label_dict={},
        ignore_models=[],
        fontsize=12,
        title_fontsize=12,
        legend_fontsize=8,
        ncols_legend=3,
        performance_label="accuracy",
        sr_label="Prob. of solution",
        fr_label="Prob. of violation",
        performance_yscale="linear",
        performance_ylims=[],
        hoz_axis_label="Amount of data",
        show_confidence_level=True,
        marker_size=20,
        save_format="pdf",
        show_title=True,
        custom_title=None,
        include_legend=True,
        savename=None,
    ):
        """Make the three plots of the experiment. Looks up any
        experiments run in self.results_dir and plots them on the 
        same three plots. 

        :param model_label_dict: An optional dictionary where keys
                are model names and values are the names you want
                shown in the legend. Note that if you specify this 
                dict, then only the models in this dictionary 
                will appear in the legend, and they will show up in the legend
                in the order that you specify them in the dict.
        :type model_label_dict: int
        :param ignore_models: Do not plot any models that appear in this list. 
        :type ignore_models: List
        :param fontsize: The font size to use for the axis labels
        :type fontsize: int
        :param legend_fontsize: The font size to use for text
                in the legend
        :type legend_fontsize: int
        :param ncols_legend: The number of columns to use in the legend
        :type ncols_legend: int, defaults to 3
        :param performance_label: The y axis label on the performance
                plot (left plot) you want to use.
        :type performance_label: str, defaults to "accuracy"
        :param sr_label: The y axis label on the solution rate 
                plot (middle plot) you want to use.
        :type sr_label: str, defaults to "Prob. of solution"
        :param fr_label: The y axis label on the failure rate 
                plot (right plot) you want to use.
        :type fr_label: str, defaults to "Prob. of violation"
        :param performance_yscale: The y axis scaling, "log" or "linear"
        :param performance_ylims: The y limits of the performance plot. 
            Default is to use matplotlib's automatic determination.
        :param hoz_axis_label: What you want to show as the horizontal axis
            label for all plots
        :type hoz_axis_label: str, defaults to "Amount of data"
        :param show_confidence_level: Whether to show the black dotted line for the value
            of delta in the failure rate plot (right plot)
        :type show_confidence_level: Bool
        :param marker_size: The size of the points in each plots
        :type marker_size: float
        :param save_format: The file type for the saved plot
        :type save_format: str, defaults to "pdf"
        :param show_title: Whether to show the title at the top of the figure
        :type show_title: bool
        :param custom_title: A custom title 
        :type custom_title: str, defaults to None
        :param include_legend: Whether to include the legend
        :type include_legend: bool, defaults to True
        :param savename: If not None, the filename to which the figure
                will be saved on disk.
        :type savename: str, defaults to None
        """
        plt.style.use("bmh")
        regime = self.spec.dataset.regime
        if regime == "supervised_learning":
            tot_data_size = self.spec.dataset.num_datapoints
        elif regime == "reinforcement_learning":
            tot_data_size = self.hyperparameter_and_setting_dict['num_episodes']

        # Read in constraints
        parse_trees = self.spec.parse_trees
        n_constraints = len(parse_trees)
        constraint_strs = [pt.constraint_str for pt in parse_trees]
        deltas = [pt.delta for pt in parse_trees]

        # Figure out what experiments we have from subfolders in results_dir
        subfolders = [
            os.path.basename(f) for f in os.scandir(self.results_dir) if f.is_dir()
        ]
        all_models = [
            x.split("_results")[0] for x in subfolders if x.endswith("_results")
        ]
        if ignore_models != []:
            all_models = [x for x in all_models if x not in ignore_models]
        seldonian_models = list(set(all_models).intersection(seldonian_model_set))
        baselines = sorted(list(set(all_models).difference(seldonian_model_set)))
        if not (seldonian_models or baselines):
            print("No results for Seldonian models or baselines found ")
            return

        ## BASELINE RESULTS SETUP
        baseline_dict = {}
        for baseline in baselines:
            baseline_dict[baseline] = {}
            savename_baseline = os.path.join(
                self.results_dir, f"{baseline}_results", f"{baseline}_results.csv"
            )
            df_baseline = pd.read_csv(savename_baseline)
            df_baseline["solution_returned"] = df_baseline["performance"].apply(
                lambda x: ~np.isnan(x)
            )
            df_baseline['gvec'] = df_baseline['gvec'].apply(lambda t: np.fromstring(t[1:-1],sep=' '))
            new_colnames = ['g' + str(ii) + '_failed' for ii in range(1,n_constraints+1)]
            for ii in range(len(new_colnames)):
                colname = new_colnames[ii]
                df_baseline[colname] = df_baseline['gvec'].str[ii].apply(has_failed)
            df_baseline = df_baseline.drop('gvec', axis=1)

            valid_mask = ~np.isnan(df_baseline["performance"])
            df_baseline_valid = df_baseline[valid_mask]
            # Get the list of all data_fracs
            X_all = df_baseline.groupby("data_frac").mean().index * tot_data_size
            # Get the list of data_fracs for which there is at least one trial that has non-nan performance
            X_valid = (
                df_baseline_valid.groupby("data_frac").mean().index * tot_data_size
            )

            baseline_dict[baseline]["df_baseline"] = df_baseline.copy()
            baseline_dict[baseline]["df_baseline_valid"] = df_baseline_valid.copy()
            baseline_dict[baseline]["X_all"] = X_all
            baseline_dict[baseline]["X_valid"] = X_valid

        # SELDONIAN RESULTS SETUP
        seldonian_dict = {}
        for seldonian_model in seldonian_models:
            seldonian_dict[seldonian_model] = {}
            savename_seldonian = os.path.join(
                self.results_dir,
                f"{seldonian_model}_results",
                f"{seldonian_model}_results.csv",
            )

            df_seldonian = pd.read_csv(savename_seldonian)
            df_seldonian['gvec'] = df_seldonian['gvec'].apply(lambda t: np.fromstring(t[1:-1],sep=' '))
            new_colnames = ['g' + str(ii) + '_failed' for ii in range(1,n_constraints+1)]
            for ii in range(len(new_colnames)):
                colname = new_colnames[ii]
                df_seldonian[colname] = df_seldonian['gvec'].str[ii].apply(has_failed)
            df_seldonian = df_seldonian.drop('gvec', axis=1)

            passed_mask = df_seldonian["passed_safety"] == True
            df_seldonian_passed = df_seldonian[passed_mask]
            # Get the list of all data_fracs
            X_all = df_seldonian.groupby("data_frac").mean().index * tot_data_size
            # Get the list of data_fracs for which there is at least one trial that passed the safety test
            X_passed = (
                df_seldonian_passed.groupby("data_frac").mean().index * tot_data_size
            )
            seldonian_dict[seldonian_model]["df_seldonian"] = df_seldonian.copy()
            seldonian_dict[seldonian_model][
                "df_seldonian_passed"
            ] = df_seldonian_passed.copy()
            seldonian_dict[seldonian_model]["X_all"] = X_all
            seldonian_dict[seldonian_model]["X_passed"] = X_passed

        ## PLOTTING SETUP
        vert_size = 3 + n_constraints
        if include_legend:
            vert_size+=0.5
            figsize = (14, vert_size)
        else:
            figsize = (14, vert_size)
        fig = plt.figure(figsize=figsize)
        plot_index = 1
        n_rows = len(constraint_strs)
        n_cols = 3
        legend_handles = []
        legend_labels = []

        # One row per constraint
        for constraint_index, constraint_str in enumerate(constraint_strs):
            constraint_num = constraint_index+1
            delta = deltas[constraint_index]

            # SETUP FOR PLOTTING
            ax_performance = fig.add_subplot(n_rows, n_cols, plot_index)
            plot_index += 1
            ax_sr = fig.add_subplot(n_rows, n_cols, plot_index, sharex=ax_performance)
            plot_index += 1
            ax_fr = fig.add_subplot(n_rows, n_cols, plot_index, sharex=ax_performance)
            plot_index += 1

            # Plot title (put above middle plot)
            if show_title:
                if custom_title:
                    title = custom_title
                else:
                    title = f"constraint: \ng={constraint_str}"
                ax_sr.set_title(title, y=1.05, fontsize=title_fontsize)

            # Plot labels
            ax_performance.set_ylabel(performance_label, fontsize=fontsize)
            ax_sr.set_ylabel(sr_label, fontsize=fontsize)
            ax_fr.set_ylabel(fr_label, fontsize=fontsize)

            # Only put horizontal axis labels on last row of plots
            if constraint_index == n_constraints - 1:
                ax_performance.set_xlabel(hoz_axis_label, fontsize=fontsize)
                ax_sr.set_xlabel(hoz_axis_label, fontsize=fontsize)
                ax_fr.set_xlabel(hoz_axis_label, fontsize=fontsize)

            # axis scaling
            ax_performance.set_xscale("log")
            if performance_yscale.lower() == "log":
                ax_performance.set_yscale("log")
            ax_sr.set_xscale("log")
            ax_fr.set_xscale("log")

            locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
            locmin = matplotlib.ticker.LogLocator(
                base=10.0,
                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                numticks=12,
            )
            for ax in [ax_performance, ax_sr, ax_fr]:
                ax.minorticks_on()
                ax.xaxis.set_major_locator(locmaj)
                ax.xaxis.set_minor_locator(locmin)
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

            ########################
            ### PERFORMANCE PLOT ###
            ########################


            # Seldonian performance
            for seldonian_i, seldonian_model in enumerate(seldonian_models):
                this_seldonian_dict = seldonian_dict[seldonian_model]
                seldonian_color = plot_colormap(seldonian_i)
                df_seldonian_passed = this_seldonian_dict["df_seldonian_passed"]
                mean_performance = df_seldonian_passed.groupby("data_frac").mean()[
                    "performance"
                ].to_numpy()
                std_performance = df_seldonian_passed.groupby("data_frac").std()[
                    "performance"
                ].to_numpy()

                n_passed = df_seldonian_passed.groupby("data_frac").count()[
                    "performance"
                ].to_numpy()
                ste_performance = std_performance / np.sqrt(n_passed)
                # Only show if 2 or more passed. Otherwise std is not defined.
                gt1_mask = n_passed > 1
                mean_performance_masked = mean_performance[gt1_mask]
                std_performance_masked = std_performance[gt1_mask]
                ste_performance_masked = ste_performance[gt1_mask]

                X_passed_seldonian = this_seldonian_dict["X_passed"].to_numpy()
                X_passed_seldonian_masked = X_passed_seldonian[gt1_mask]
                (pl,) = ax_performance.plot(
                    X_passed_seldonian_masked,
                    mean_performance_masked,
                    color=seldonian_color,
                    # linestyle="--",
                    linestyle="-",
                )
                if constraint_index == 0:
                    legend_handles.append(pl)
                    if seldonian_model in model_label_dict:
                        legend_labels.append(model_label_dict[seldonian_model])
                    else:
                        legend_labels.append(seldonian_model)

                ax_performance.scatter(
                    X_passed_seldonian_masked,
                    mean_performance_masked,
                    color=seldonian_color,
                    s=marker_size,
                    marker="o",
                    zorder=10
                )
                ax_performance.fill_between(
                    X_passed_seldonian_masked,
                    mean_performance_masked - ste_performance_masked,
                    mean_performance_masked + ste_performance_masked,
                    color=seldonian_color,
                    alpha=0.5,
                    zorder=10
                )
            
            # Baseline performance
            for baseline_i, baseline in enumerate(baselines):
                baseline_color = plot_colormap(
                    baseline_i + len(seldonian_models)
                )  # 0 is reserved for Seldonian model
                this_baseline_dict = baseline_dict[baseline]
                df_baseline_valid = this_baseline_dict["df_baseline_valid"]
                n_trials = df_baseline_valid["trial_i"].max() + 1

                # Performance
                baseline_mean_performance = df_baseline_valid.groupby(
                    "data_frac"
                ).mean()["performance"]
                baseline_std_performance = df_baseline_valid.groupby("data_frac").std()[
                    "performance"
                ]
                baseline_ste_performance = baseline_std_performance / np.sqrt(n_trials)
                X_valid_baseline = this_baseline_dict["X_valid"]
                n_valid = df_baseline_valid.groupby("data_frac").count()[
                     "performance"
                 ].to_numpy()
                # Only show if 2 or more passed. Otherwise std is not defined.
                gt1_mask = n_valid > 1
                baseline_mean_performance_masked = baseline_mean_performance[gt1_mask]
                baseline_std_performance_masked = baseline_std_performance[gt1_mask]
                baseline_ste_performance_masked = baseline_ste_performance[gt1_mask]
                X_valid_baseline_masked = X_valid_baseline[gt1_mask]
                (pl,) = ax_performance.plot(
                    X_valid_baseline_masked.to_numpy(),
                    baseline_mean_performance_masked.to_numpy(),
                    color=baseline_color,
                    label=baseline,
                )
                if constraint_index == 0:
                    legend_handles.append(pl)
                    if baseline in model_label_dict:
                        legend_labels.append(model_label_dict[baseline])
                    else:
                        legend_labels.append(baseline)
                ax_performance.scatter(
                    X_valid_baseline_masked,
                    baseline_mean_performance_masked,
                    color=baseline_color,
                    s=marker_size,
                    marker=marker_list[baseline_i],
                )
                ax_performance.fill_between(
                    X_valid_baseline_masked,
                    baseline_mean_performance_masked - baseline_ste_performance_masked,
                    baseline_mean_performance_masked + baseline_ste_performance_masked,
                    color=baseline_color,
                    alpha=0.5,
                )
           
            if performance_ylims:
                ax_performance.set_ylim(*performance_ylims)
            ##########################
            ### SOLUTION RATE PLOT ###
            ##########################

            # Seldonian solution rate
            for seldonian_i, seldonian_model in enumerate(seldonian_models):
                this_seldonian_dict = seldonian_dict[seldonian_model]
                seldonian_color = plot_colormap(seldonian_i)
                df_seldonian = this_seldonian_dict["df_seldonian"]
                n_trials = df_seldonian["trial_i"].max() + 1
                mean_sr = df_seldonian.groupby("data_frac").mean()["passed_safety"].to_numpy()
                std_sr = df_seldonian.groupby("data_frac").std()["passed_safety"].to_numpy()
                ste_sr = std_sr / np.sqrt(n_trials)

                X_all_seldonian = this_seldonian_dict["X_all"].to_numpy()

                ax_sr.plot(
                    X_all_seldonian,
                    mean_sr,
                    color=seldonian_color,
                    # linestyle="--",
                    linestyle="-",
                    label="QSA",
                    zorder=10
                )
                ax_sr.scatter(
                    X_all_seldonian,
                    mean_sr,
                    color=seldonian_color,
                    s=marker_size,
                    marker="o",
                    zorder=10
                )
                ax_sr.fill_between(
                    X_all_seldonian,
                    mean_sr - ste_sr,
                    mean_sr + ste_sr,
                    color=seldonian_color,
                    alpha=0.5,
                    zorder=10
                )

            # Plot baseline solution rate
            # (sometimes it doesn't return a solution due to not having enough training data
            # to run model.fit() )
            for baseline_i, baseline in enumerate(baselines):
                this_baseline_dict = baseline_dict[baseline]
                baseline_color = plot_colormap(baseline_i + len(seldonian_models))
                df_baseline = this_baseline_dict["df_baseline"]
                n_trials = df_baseline["trial_i"].max() + 1
                mean_sr = df_baseline.groupby("data_frac").mean()["solution_returned"].to_numpy()
                std_sr = df_baseline.groupby("data_frac").std()["solution_returned"].to_numpy()
                ste_sr = std_sr / np.sqrt(n_trials)

                X_all_baseline = this_baseline_dict["X_all"].to_numpy()

                ax_sr.plot(
                    X_all_baseline, mean_sr, color=baseline_color, label=baseline
                )
                ax_sr.scatter(
                    X_all_baseline,
                    mean_sr,
                    color=baseline_color,
                    s=marker_size,
                    marker=marker_list[baseline_i],
                )
                ax_sr.fill_between(
                    X_all_baseline,
                    mean_sr - ste_sr,
                    mean_sr + ste_sr,
                    color=baseline_color,
                    alpha=0.5,
                )

            ax_sr.set_ylim(-0.05, 1.05)

            ##########################
            ### FAILURE RATE PLOT ###
            ##########################

            # Seldonian failure rate
            for seldonian_i, seldonian_model in enumerate(seldonian_models):
                this_seldonian_dict = seldonian_dict[seldonian_model]
                seldonian_color = plot_colormap(seldonian_i)
                df_seldonian = this_seldonian_dict["df_seldonian"]
                n_trials = df_seldonian["trial_i"].max() + 1

                gstr_failed = 'g' + str(constraint_num) + '_failed'

                mean_fr = df_seldonian.groupby("data_frac").mean()[
                    gstr_failed].to_numpy()
                # Need to groupby data frac
                std_fr = df_seldonian.groupby("data_frac").std()[
                    gstr_failed].to_numpy()

                ste_fr = std_fr / np.sqrt(n_trials)

                X_all_seldonian = this_seldonian_dict["X_all"].to_numpy()

                ax_fr.plot(
                    X_all_seldonian,
                    mean_fr,
                    color=seldonian_color,
                    # linestyle="--",
                    linestyle="-",
                    label="QSA",
                    zorder=10
                )
                ax_fr.fill_between(
                    X_all_seldonian,
                    mean_fr - ste_fr,
                    mean_fr + ste_fr,
                    color=seldonian_color,
                    alpha=0.5,
                    zorder=10
                )
                ax_fr.scatter(
                    X_all_seldonian,
                    mean_fr,
                    color=seldonian_color,
                    s=marker_size,
                    marker="o",
                    zorder=10
                )

            # Baseline failure rate
            for baseline_i, baseline in enumerate(baselines):
                baseline_color = plot_colormap(baseline_i + len(seldonian_models))
                # Baseline performance
                this_baseline_dict = baseline_dict[baseline]
                df_baseline = this_baseline_dict["df_baseline"]
                n_trials = df_baseline["trial_i"].max() + 1

                # baseline_mean_fr = df_baseline_valid.groupby("data_frac").mean()[
                #     "failed"
                # ].to_numpy()
                # baseline_std_fr = df_baseline_valid.groupby("data_frac").std()["failed"].to_numpy()
                gstr_failed = 'g' + str(constraint_num) + '_failed'

                baseline_mean_fr = df_baseline.groupby("data_frac").mean()[
                    gstr_failed].to_numpy()
                # Need to groupby data frac
                baseline_std_fr = df_baseline.groupby("data_frac").std()[
                    gstr_failed].to_numpy()
                baseline_ste_fr = baseline_std_fr / np.sqrt(n_trials)

                X_all_baseline = this_baseline_dict["X_all"].to_numpy()

                ax_fr.plot(
                    X_all_baseline,
                    baseline_mean_fr,
                    color=baseline_color,
                    label=baseline,
                )
                ax_fr.scatter(
                    X_all_baseline,
                    baseline_mean_fr,
                    color=baseline_color,
                    marker=marker_list[baseline_i],
                    s=marker_size,
                )
                ax_fr.fill_between(
                    X_all_baseline,
                    baseline_mean_fr - baseline_ste_fr,
                    baseline_mean_fr + baseline_ste_fr,
                    color=baseline_color,
                    alpha=0.5,
                )

            ax_fr.set_ylim(-0.05, 1.05)
            if show_confidence_level:
                ax_fr.axhline(
                    y=delta, color="k", linestyle="--", label=f"delta={delta}"
                )
        plt.tight_layout()

        if include_legend:
            if model_label_dict:
                reordered_legend_labels = []
                reordered_legend_handles = []
                for name in model_label_dict:
                    display_name = model_label_dict[name]
                    if display_name in legend_labels:
                        leg_index = legend_labels.index(display_name)
                        leg_name = legend_labels[leg_index]
                        leg_handle = legend_handles[leg_index]
                        reordered_legend_labels.append(leg_name)
                        reordered_legend_handles.append(leg_handle)
                legend_handles = reordered_legend_handles
                legend_labels = reordered_legend_labels
            fig.subplots_adjust(bottom=0.25)
            fig.legend(
                legend_handles,
                legend_labels,
                bbox_to_anchor=(0.5, 0.15),
                loc="upper center",
                ncol=ncols_legend,
                fontsize=legend_fontsize,
            )

        if savename:
            plt.savefig(savename, format=save_format,bbox_inches="tight")
            print(f"Saved {savename}")
        else:
            plt.show()


    def get_selected_frac_data_in_safety(
            self,
    ):
        """
        For each trial and datafrac, gets information about the selected safety_frac used.

        :return: selected_rho_dict, maps data_frac to a list of selected rhos for each trial
        :rtype: dict
        """
        selected_rho_dict = {}

        for data_frac in self.data_fracs:
            selected_rho_dict[data_frac] = []
            for trial_i in range(self.n_trials):
                all_bootstrap_est_path = os.path.join(
                        self.results_dir, "all_bootstrap_info", 
                        f"bootstrap_info_{trial_i}_{data_frac:.4f}", "all_bootstrap_est.csv")
                bootstrap_est_df = pd.read_csv(all_bootstrap_est_path)
                if bootstrap_est_df.empty:
                    selected_rho_dict[data_frac].append(0)
                else:
                    selected_rho_dict[data_frac].append(
                            min(bootstrap_est_df["frac_data_in_safety"]))

        return selected_rho_dict


    def get_all_bootstrap_trial_info(
            self,
            all_safety_frac
    ):
        """
        Load information from all bootstrap trials, i.e. for all datafracs, for all
            safety_frac and predicted future_safety frac, get result from every bootstrap
            trial ran.


        Returns all_bootstrap_trial_info, a dictionary such that
            all_bootstrap_trial_info[safety_frac][future_safety_frac] is a 
            (self.n_trials, len(self.datafracs), self.n_bootstrap_trials) array storing
            the results from each bootstrap trial for each trial.
        """
        tot_data_size = self.spec.dataset.num_datapoints
        all_safety_frac.sort(reverse=True) # Sort largest to smallest.

        all_bootstrap_trial_info = {safety_frac: {} for safety_frac in all_safety_frac}

        for safety_frac in all_safety_frac:
            for future_safety_frac in all_safety_frac:
                if (future_safety_frac > safety_frac): continue
                all_bootstrap_trial_info[safety_frac][future_safety_frac] = \
                        np.zeros((self.n_trials, len(self.data_fracs), self.n_bootstrap_trials))

                for trial_i in range(self.n_trials):
                    for data_frac_i, data_frac in enumerate(self.data_fracs):
                        future_safety_frac_csv_path = os.path.join(
                                self.results_dir, "all_bootstrap_info", 
                                f"bootstrap_info_{trial_i}_{data_frac:.4f}",
                                f"bootstrap_safety_frac_{safety_frac:.2f}",
                                f"future_safety_frac_{future_safety_frac:.2f}",
                                "all_bs_trials_results.csv")

                        if os.path.exists(future_safety_frac_csv_path):
                            future_safety_frac_csv = pd.read_csv(future_safety_frac_csv_path)

                            all_bootstrap_trial_info[safety_frac][future_safety_frac][
                                    trial_i, data_frac_i, :] = future_safety_frac_csv["passed_safety"]

                            """
                            all_probpass_est[safety_frac][future_safety_frac][trial_i,
                                data_frac_i] = np.nanmean(future_safety_frac_csv["passed_safety"])
                            all_trial_std[safety_frac][future_safety_frac][trial_i, data_frac_i] = np.nanstd(future_safety_frac_csv["passed_safety"])
                            all_pass_count[safety_frac][future_safety_frac][trial_i, data_frac_i] = np.sum(future_safety_frac_csv["passed_safety"])
                            """
                        else: # not run for this data_frac, safety_frac and future_safety_frac
                            all_bootstrap_trial_info[safety_frac][future_safety_frac][
                                    trial_i, data_frac_i, :] = np.nan

        return all_bootstrap_trial_info


    def clopper_pearson_bound(
            self,
            pass_count, 
            num_bootstrap_samples,
            alpha=0.05, # Acceptable error
    ):
        """
        Computes a 1-alpha clopper pearson bound on the probability of passing. 

        :param pass_count: array of containing an entry for each corresponding 
            datafrac, containing the count of number of 
        :type pass_count: np.array
        :param num_bootstrap_samples: number of bootstrap samples used to compute estimates
            of passing (number of draws from the binomial in bound)
        :type num_bootstrap_samples: int
        :param alpha: confidence parameter
        :type num_bootstrap_samples: float
        """
        lower_bound = scipy.stats.beta.ppf(alpha/2, pass_count, num_bootstrap_samples - 
                pass_count + 1)
        upper_bound = scipy.stats.beta.ppf(1 - alpha/2, pass_count+ 1, 
                num_bootstrap_samples - pass_count)

        return lower_bound, upper_bound


    def ttest_bound(
            self,
            bootstrap_trial_data,
            delta=0.1
    ):
        """
        Computes upper and lower ttest bounds for the probability of passing across
            bootstrap trials. Variance is across bootstrap trials, not pool.

        :param bootstrap_trial_data: (len(self.data_fracs), self.n_bootstrap_trials) array
            containing the result for each bootstrap trial for the given data_frac
        :type bootstrap_trial_data: np.array
        :param delta: confidence level, i.e. 0.05
        :type delta: float
        """
        # Compute mea and standard deviation across bootstrap trials
        bootstrap_data_mean = np.nanmean(bootstrap_trial_data, axis=1)
        bootstrap_data_stddev = np.nanstd(bootstrap_trial_data, axis=1)

        lower_bound = bootstrap_data_mean - bootstrap_data_stddev / np.sqrt(
                self.n_bootstrap_trials) * tinv(1.0 - delta, self.n_bootstrap_trials - 1)
        upper_bound = bootstrap_data_mean + bootstrap_data_stddev / np.sqrt(
                self.n_bootstrap_trials) * tinv(1.0 - delta, self.n_bootstrap_trials- 1)

        return lower_bound, upper_bound


    def make_probpass_est_plot(
        self,
        all_safety_frac,
        bound_type="",
    ):
        """
        Creates plot for each safety_frac value in all_safety_frac, of the estimated probability
            of passing for each considered future_safety_frac. Estimates are computed across trial.

        :param all_safety_frac: array of containing all values of safety frac being considered
        :type pass_count: np.array
        :param bound_type: indicates what time of confidence bound to put around
        :type bound_type: string
        """
        tot_data_size = self.spec.dataset.num_datapoints
        all_data_size = np.array(self.data_fracs) * tot_data_size
        all_bootstrap_trial_info = self.get_all_bootstrap_trial_info(all_safety_frac)

        for safety_frac in all_safety_frac:

            # Make plot across average prob pass for each rho, averaged across each trial.
            plt.figure()
            for future_safety_frac in all_safety_frac:
                if (future_safety_frac > safety_frac): continue # Only estimate smaller safety_frac

                # Compute the probability estimates computed for each trial.
                all_trial_probpass_est = np.nanmean(all_bootstrap_trial_info[
                    safety_frac][future_safety_frac], axis=2) # Average over bootstrap trials.

                # Note: this is computing mean and std over experiment trials, not bootstrap trials.
                mean_probpass_est = np.nanmean(all_trial_probpass_est, axis=0)
                ste_probpass_est = np.nanstd(all_trial_probpass_est, axis=0) / np.sqrt(self.n_trials)

                plt.plot(all_data_size, mean_probpass_est, label=future_safety_frac)
                plt.fill_between(all_data_size, mean_probpass_est - ste_probpass_est, 
                        mean_probpass_est + ste_probpass_est, alpha=0.1)

            plt.xscale("log")
            plt.title(f"rho {safety_frac:.2f}, mean")
            plt.legend(title="future safety rho")


    def make_trial_probpass_est_plot(
        self,
        all_safety_frac,
        bound_type,
    ):
        """
        Creates plot of the estimated probability of passing for each future_safety_frac
            for the given safety_frac

        :param all_safety_frac: array of containing all values of safety frac being considered
        :type pass_count: np.array
        :param bound_type: indicates what time of confidence bound to put around
        :type bound_type: string
        """
        tot_data_size = self.spec.dataset.num_datapoints
        all_data_size = np.array(self.data_fracs) * tot_data_size
        all_bootstrap_trial_info = self.get_all_bootstrap_trial_info(all_safety_frac)

        for safety_frac in all_safety_frac:
            for trial_i in range(self.n_trials):

                plt.figure()
                for future_safety_frac in all_safety_frac:
                    if (future_safety_frac > safety_frac): continue # Only estimate smaller safety_frac

                    # Average over bootstrap trials, to get the estimate for trial_i.
                    trial_probpass_est = np.nanmean(all_bootstrap_trial_info[
                        safety_frac][future_safety_frac][trial_i], axis=1) 
                    plt.plot(all_data_size, trial_probpass_est, label=future_safety_frac)


                    if bound_type == "ste": 
                        # Get the standard error of the Bernoulli of the bootstrap trials.
                        bootstrap_trial_ste = np.nanstd(all_bootstrap_trial_info[
                            safety_frac][future_safety_frac][trial_i], axis=1) / np.sqrt(self.n_bootstrap_trials)
                        plt.fill_between(all_data_size, trial_probpass_est - bootstrap_trial_ste, 
                                trial_probpass_est + bootstrap_trial_ste, alpha=0.1)

                    elif bound_type == "clopper-pearson":
                        trial_pass_count = np.nansum(all_bootstrap_trial_info[
                            safety_frac][future_safety_frac][trial_i], axis=1)
                        lower_bound, upper_bound = self.clopper_pearson_bound(
                                trial_pass_count, num_bootstrap_trials)
                        lower_bound = np.nan_to_num(lower_bound) # If lower bound nan, set to 0.
                        plt.fill_between(all_data_size, lower_bound, upper_bound, alpha=0.1)

                    elif bound_type == "ttest":
                        lower_bound, upper_bound = self.ttest_bound(all_bootstrap_trial_info[
                            safety_frac][future_safety_frac][trial_i])
                        plt.fill_between(all_data_size, lower_bound, upper_bound, alpha=0.1)

                plt.xscale("log")
                plt.title(f"rho {safety_frac:.2f}, trial {trial_i}")
                plt.legend(title="future safety rho")


    def make_selected_safety_frac_plot(
        self,
        model_label_dict={},
        fontsize=12,
        legend_fontsize=8,
        performance_label="accuracy",
        performance_xscale="linear",
        performance_yscale="linear",
        performance_xlims=[],
        performance_ylims=[],
        marker_size=20,
        save_format="pdf",
        show_title=True,
        custom_title=None,
        include_legend=True,
        savename=None,
    ):
        """
        Plot the selected rho values across dataset size.
        """
        tot_data_size = self.spec.dataset.num_datapoints

        # Get all the rhos selected across trials.
        selected_rho_dict = self.get_selected_frac_data_in_safety()
        all_data_frac = list(selected_rho_dict.keys())
        print(all_data_frac)
        all_data_size = np.array(all_data_frac) * tot_data_size

        # Convert to scatter data and average data.
        scatter_data_frac, scatter_rho = [], []
        mean_rho = []
        ste_rho = []
        for data_frac in selected_rho_dict.keys():
            for trial in range(len(selected_rho_dict[data_frac])):
                scatter_data_frac.append(data_frac * tot_data_size)
                scatter_rho.append(selected_rho_dict[data_frac][trial])
            mean_rho.append(np.mean(selected_rho_dict[data_frac]))
            std_rho = np.std(selected_rho_dict[data_frac])
            ste_rho.append(std_rho/ np.sqrt(self.n_trials))
        mean_rho = np.array(mean_rho)
        ste_rho = np.array(ste_rho)


        plt.scatter(scatter_data_frac, scatter_rho, alpha=0.2)
        plt.plot(all_data_size, mean_rho)
        plt.fill_between(all_data_size, mean_rho - ste_rho, mean_rho + std_rho, alpha=0.1)
        plt.xscale("log")
        plt.ylim(0, 1.0)

        print(selected_rho_dict)



    def make_all_safety_frac_probpass_plots(
        self,
        model_label_dict={},
        fontsize=12,
        legend_fontsize=8,
        performance_label="accuracy",
        performance_xscale="linear",
        performance_yscale="linear",
        performance_xlims=[],
        performance_ylims=[],
        marker_size=20,
        save_format="pdf",
        show_title=True,
        custom_title=None,
        include_legend=True,
        savename=None,
    ):
        plt.style.use("bmh")
        # plt.style.use('grayscale')
        regime = self.spec.dataset.regime
        tot_data_size = self.spec.dataset.num_datapoints

        # Read in constraints
        parse_trees = self.spec.parse_trees

        constraint_dict = {}
        for pt_ii, pt in enumerate(parse_trees):
            delta = pt.delta
            constraint_str = pt.constraint_str
            constraint_dict[f"constraint_{pt_ii}"] = {
                "delta": delta,
                "constraint_str": constraint_str,
            }

        constraints = list(constraint_dict.keys())

        
        # Figure out what experiments we have from subfolders in results_dir
        subfolders = [
            os.path.basename(f) for f in os.scandir(self.results_dir) if f.is_dir()
        ]
        all_models = [
            x.split("_results")[0] for x in subfolders if x.endswith("_results")
        ]
        seldonian_models = list(set(all_models).intersection(seldonian_model_set))
        baselines = sorted(list(set(all_models).difference(seldonian_model_set)))
        if not (seldonian_models or baselines):
            print("No results for Seldonian models or baselines found ")
            return

        # Load data for plotting fixed rho performance plots.
        safety_frac_dict = load_pickle(os.path.join(self.results_dir, "fixed_rho_plot_data.csv"))
        print(safety_frac_dict)
        all_safety_frac = safety_frac_dict.keys()

        # SELECT SAFETY FRAC RESULTS SETUP
        seldonian_dict = {}
        for seldonian_model in seldonian_models: 
            seldonian_dict[seldonian_model] = {}
            savename_seldonian = os.path.join(
                self.results_dir,
                f"{seldonian_model}_results",
                f"{seldonian_model}_results.csv",
            )

            df_seldonian = pd.read_csv(savename_seldonian)
            passed_mask = df_seldonian["passed_safety"] == True
            df_seldonian_passed = df_seldonian[passed_mask]
            # Get the list of all data_fracs
            X_all = df_seldonian.groupby("data_frac").mean().index * tot_data_size
            # Get the list of data_fracs for which there is at least one trial that passed the safety test
            X_passed = (
                df_seldonian_passed.groupby("data_frac").mean().index * tot_data_size
            )
            seldonian_dict[seldonian_model]["df_seldonian"] = df_seldonian.copy()
            seldonian_dict[seldonian_model][
                "df_seldonian_passed"
            ] = df_seldonian_passed.copy()
            seldonian_dict[seldonian_model]["X_all"] = X_all
            seldonian_dict[seldonian_model]["X_passed"] = X_passed


        ## PLOTTING SETUP
        fig = plt.figure()
        ax_sr = fig.add_subplot(111)
        ax_sr.set_xlabel("Amount of data", fontsize=fontsize)
        ax_sr.set_ylabel("Probability of solution", fontsize=fontsize)
        ax_sr.set_xscale("log")
        if performance_xlims: ax_sr.set_xlim(*performance_xlims)
        if performance_ylims: ax_sr.set_ylim(*performance_ylims)
        legend_handles = []
        legend_labels = []


        ## Loop over constraints and make plots for each constraint
        for ii, constraint in enumerate(constraints):
            constraint_str = constraint_dict[constraint]["constraint_str"]
            delta = constraint_dict[constraint]["delta"]

            ##########################
            ### SOLUTION RATE PLOT ###
            ##########################
            for safety_frac_i, safety_frac in enumerate(all_safety_frac):

                # Fixed rho plots.
                this_safety_frac_dict = safety_frac_dict[safety_frac]
                safety_frac_color = plot_colormap(safety_frac_i)
                df_safety_frac = this_safety_frac_dict["df_safetyfrac"]
                n_trials = df_safety_frac["trial_i"].max() + 1
                mean_sr = df_safety_frac.groupby("data_frac").mean()["passed_safety"]
                std_sr = df_safety_frac.groupby("data_frac").std()["passed_safety"]
                ste_sr = std_sr / np.sqrt(n_trials)

                X_all_safety_frac = this_safety_frac_dict["X_all"]

                ax_sr.plot(
                    X_all_safety_frac,
                    mean_sr,
                    color=safety_frac_color,
                    linestyle="--",
                    label=f"{safety_frac:.1f}",
                )
                ax_sr.scatter(
                    X_all_safety_frac,
                    mean_sr,
                    color=safety_frac_color,
                    s=marker_size,
                    marker="o",
                )
                ax_sr.fill_between(
                    X_all_safety_frac,
                    mean_sr - ste_sr,
                    mean_sr + ste_sr,
                    color=safety_frac_color,
                    alpha=0.2,
                )

            # These are the selected models
            for seldonian_i, seldonian_model in enumerate(seldonian_models):
                this_seldonian_dict = seldonian_dict[seldonian_model]
                seldonian_color = plot_colormap(seldonian_i + len(all_safety_frac))
                df_seldonian = this_seldonian_dict["df_seldonian"]
                n_trials = df_seldonian["trial_i"].max() + 1
                mean_sr = df_seldonian.groupby("data_frac").mean()["passed_safety"]
                std_sr = df_seldonian.groupby("data_frac").std()["passed_safety"]
                ste_sr = std_sr / np.sqrt(n_trials)

                X_all_seldonian = this_seldonian_dict["X_all"]

                ax_sr.plot(
                    X_all_seldonian,
                    mean_sr,
                    color=seldonian_color,
                    linestyle="--",
                    label="select rho",
                )
                ax_sr.scatter(
                    X_all_seldonian,
                    mean_sr,
                    color=seldonian_color,
                    s=marker_size,
                    marker="o",
                )
                ax_sr.fill_between(
                    X_all_seldonian,
                    mean_sr - ste_sr,
                    mean_sr + ste_sr,
                    color=seldonian_color,
                    alpha=0.5,
                )

            ax_sr.set_ylim(-0.05, 1.05)


        fig.legend(
                bbox_to_anchor=(0.5, 0),
                loc="upper center",
                ncol=5,
                title=r"$\rho$"
        )

        if savename:
            plt.savefig(savename,format=save_format,dpi=600)
            # plt.savefig(savename, format=save_format)
            print(f"Saved {savename}")
        else:
            plt.show()



class SupervisedPlotGenerator(PlotGenerator):
    def __init__(
        self,
        spec,
        n_trials,
        data_fracs,
        datagen_method,
        perf_eval_fn,
        results_dir,
        n_workers,
        n_bootstrap_trials=None,
        hyperparam_select_spec=None,
        constraint_eval_fns=[],
        perf_eval_kwargs={},
        constraint_eval_kwargs={},
        batch_epoch_dict={},
    ):
        """Class for running supervised Seldonian experiments
                and generating the three plots

        :param spec: Specification object for running the
                Seldonian algorithm
        :type spec: seldonian.spec.Spec object

        :param n_trials: The number of times the
                Seldonian algorithm is run for each data fraction.
                Used for generating error bars
        :type n_trials: int

        :param data_fracs: Proportions of the overall size
                of the dataset to use
                (the horizontal axis on the three plots).
        :type data_fracs: List(float)

        :param datagen_method: Method for generating data that is used
                to run the Seldonian algorithm for each trial
        :type datagen_method: str, e.g. "resample"

        :param perf_eval_fn: Function used to evaluate the performance
                of the model obtained in each trial, with signature:
                func(theta,**kwargs), where theta is the solution
                from candidate selection
        :type perf_eval_fn: function or class method

        :param results_dir: The directory in which to save the results
        :type results_dir: str

        :param n_workers: The number of workers to use if
                using multiprocessing
        :type n_workers: int

        :param hyperparam_select_spec: Specification object for doing hyperparameter
                selection
        :type hyperparam_select_spec: seldonian.spec.HyperparameterSelectionSpec object

        :param constraint_eval_fns: List of functions used to evaluate
                the constraints on ground truth. If an empty list is provided,
                the constraints are evaluated using the parse tree
        :type constraint_eval_fns: List(function or class method),
                defaults to []

        :param perf_eval_kwargs: Extra keyword arguments to pass to
                perf_eval_fn
        :type perf_eval_kwargs: dict

        :param constraint_eval_kwargs: Extra keyword arguments to pass to
                the constraint_eval_fns
        :type constraint_eval_kwargs: dict

        :param batch_epoch_dict: Instruct batch sizes and n_epochs
                for each data frac
        :type batch_epoch_dict: dict
        """

        super().__init__(
            spec=spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            results_dir=results_dir,
            n_workers=n_workers,
            n_bootstrap_trials=n_bootstrap_trials,
            hyperparam_select_spec=hyperparam_select_spec,
            constraint_eval_fns=constraint_eval_fns,
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs,
            batch_epoch_dict=batch_epoch_dict,
        )
        self.regime = "supervised_learning"
    
    def run_seldonian_experiment(self, verbose=False):
        """Run a supervised Seldonian experiment using the spec attribute
        assigned to the class in __init__().

        :param verbose: Whether to display results to stdout
                while the Seldonian algorithms are running in each trial
        :type verbose: bool, defaults to False
        """

        dataset = self.spec.dataset

        if self.datagen_method == "resample":
            # Generate n_trials resampled datasets of full length
            # These will be cropped to data_frac fractional size
            print("generating resampled datasets")
            generate_resampled_datasets(dataset, self.n_trials, self.results_dir)
            print("Done generating resampled datasets")
            print()

        run_seldonian_kwargs = dict(
            spec=self.spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
            hyperparam_select_spec=self.hyperparam_select_spec,
            datagen_method=self.datagen_method,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            batch_epoch_dict=self.batch_epoch_dict,
            verbose=verbose,
        )

        ## Run experiment
        sd_exp = SeldonianExperiment(model_name="qsa", results_dir=self.results_dir)

        sd_exp.run_experiment(**run_seldonian_kwargs)
        return

    def run_headless_seldonian_experiment(
        self, 
        full_pretraining_model,
        initial_state_dict, 
        headless_pretraining_model, 
        head_layer_names,
        latent_feature_shape,
        loss_func_pretraining,
        learning_rate_pretraining,
        pretraining_device,
        batch_epoch_dict_pretraining={},
        safety_batch_size_pretraining=1000,
        verbose=False):
        """Run a headless supervised Seldonian experiment using the spec attribute
        assigned to the class in __init__().

        :param verbose: Whether to display results to stdout
                while the Seldonian algorithms are running in each trial
        :type verbose: bool, defaults to False
        """

        dataset = self.spec.dataset

        if self.datagen_method == "resample":
            # Generate n_trials resampled datasets of full length
            # These will be cropped to data_frac fractional size
            print("generating resampled datasets")
            generate_resampled_datasets(dataset, self.n_trials, self.results_dir)
            print("Done generating resampled datasets")
            print()
        else:
            raise NotImplementedError(
                f"datagen_method {datagen_method} not supported for headless experiments")

        run_kwargs = dict(
            spec=self.spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            full_pretraining_model=full_pretraining_model,
            initial_state_dict=initial_state_dict,
            headless_pretraining_model=headless_pretraining_model,
            head_layer_names=head_layer_names,
            latent_feature_shape=latent_feature_shape,
            batch_epoch_dict_pretraining=batch_epoch_dict_pretraining,
            safety_batch_size_pretraining=safety_batch_size_pretraining,
            loss_func_pretraining=loss_func_pretraining,
            learning_rate_pretraining=learning_rate_pretraining,
            pretraining_device=pretraining_device,
            n_workers=self.n_workers,
            datagen_method=self.datagen_method,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            batch_epoch_dict=self.batch_epoch_dict,
            verbose=verbose,
        )
        from .headless_experiments import HeadlessSeldonianExperiment
        ## Run experiment
        sd_exp = HeadlessSeldonianExperiment(
            model_name="headless_qsa",
            results_dir=self.results_dir)

        sd_exp.run_experiment(**run_kwargs)
        return

    def run_baseline_experiment(self, baseline_model, verbose=False):
        """Run a supervised Seldonian experiment using the spec attribute
        assigned to the class in __init__().

        :param verbose: Whether to display results to stdout
                while the Seldonian algorithms are running in each trial
        :type verbose: bool, defaults to False
        """

        dataset = self.spec.dataset

        if self.datagen_method == "resample":
            # Generate n_trials resampled datasets of full length
            # These will be cropped to data_frac fractional size
            print("checking for resampled datasets")
            generate_resampled_datasets(dataset, self.n_trials, self.results_dir)
            print("Done checking for resampled datasets")
            print()

        run_baseline_kwargs = dict(
            spec=self.spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
            datagen_method=self.datagen_method,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            hyperparam_select_spec=self.hyperparam_select_spec,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            batch_epoch_dict=self.batch_epoch_dict,
            verbose=verbose,
        )

        ## Run experiment
        bl_exp = BaselineExperiment(baseline_model=baseline_model, results_dir=self.results_dir)

        bl_exp.run_experiment(**run_baseline_kwargs)
        return

    def run_fairlearn_experiment(
        self,
        fairlearn_sensitive_feature_names,
        fairlearn_constraint_name,
        fairlearn_epsilon_constraint,
        fairlearn_epsilon_eval,
        fairlearn_eval_kwargs={},
        verbose=False,
    ):
        """Run a supervised experiment using the fairlearn
        library

        :param verbose: Whether to display results to stdout
                while the fairlearn algorithms are running in each trial
        :type verbose: bool, defaults to False
        """

        dataset = self.spec.dataset

        if self.datagen_method == "resample":
            # Generate n_trials resampled datasets of full length
            # These will be cropped to data_frac fractional size
            print("Checking for resampled datasets")
            generate_resampled_datasets(
                dataset,
                self.n_trials,
                self.results_dir,
            )
            print("Done generating resampled datasets")
            print()

        run_fairlearn_kwargs = dict(
            spec=self.spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
            datagen_method=self.datagen_method,
            fairlearn_sensitive_feature_names=fairlearn_sensitive_feature_names,
            fairlearn_constraint_name=fairlearn_constraint_name,
            fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
            fairlearn_epsilon_eval=fairlearn_epsilon_eval,
            fairlearn_eval_kwargs=fairlearn_eval_kwargs,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            verbose=verbose,
        )

        ## Run experiment
        fl_exp = FairlearnExperiment(
            results_dir=self.results_dir,
            fairlearn_epsilon_constraint=fairlearn_epsilon_constraint,
        )

        fl_exp.run_experiment(**run_fairlearn_kwargs)
        return


# TODO: add data fraction selection here
class RLPlotGenerator(PlotGenerator):
    def __init__(
        self,
        spec,
        n_trials,
        data_fracs,
        datagen_method,
        hyperparameter_and_setting_dict,
        perf_eval_fn,
        results_dir,
        n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs={},
        constraint_eval_kwargs={},
        batch_epoch_dict={},
    ):
        """Class for running RL Seldonian experiments
                and generating the three plots

        :param spec: Specification object for running the
                Seldonian algorithm
        :type spec: seldonian.spec.Spec object

        :param n_trials: The number of times the
                Seldonian algorithm is run for each data fraction.
                Used for generating error bars
        :type n_trials: int

        :param data_fracs: Proportions of the overall size
                of the dataset to use
                (the horizontal axis on the three plots).
        :type data_fracs: List(float)

        :param datagen_method: Method for generating data that is used
                to run the Seldonian algorithm for each trial
        :type datagen_method: str, e.g. "resample"

        :param perf_eval_fn: Function used to evaluate the performance
                of the model obtained in each trial, with signature:
                func(theta,**kwargs), where theta is the solution
                from candidate selection
        :type perf_eval_fn: function or class method

        :param results_dir: The directory in which to save the results
        :type results_dir: str

        :param n_workers: The number of workers to use if
                using multiprocessing
        :type n_workers: int

        :param constraint_eval_fns: List of functions used to evaluate
                the constraints on ground truth. If an empty list is provided,
                the constraints are evaluated using the parse tree
        :type constraint_eval_fns: List(function or class method),
                defaults to []

        :param perf_eval_kwargs: Extra keyword arguments to pass to
                perf_eval_fn
        :type perf_eval_kwargs: dict

        :param constraint_eval_kwargs: Extra keyword arguments to pass to
                the constraint_eval_fns
        :type constraint_eval_kwargs: dict

        :param batch_epoch_dict: Instruct batch sizes and n_epochs
                for each data frac
        :type batch_epoch_dict: dict
        """

        super().__init__(
            spec=spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            results_dir=results_dir,
            n_workers=n_workers,
            constraint_eval_fns=constraint_eval_fns,
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs,
            batch_epoch_dict=batch_epoch_dict,
        )

        self.regime = "reinforcement_learning"
        self.hyperparameter_and_setting_dict = hyperparameter_and_setting_dict

    def run_seldonian_experiment(self, verbose=False):
        """Run an RL Seldonian experiment using the spec attribute
        assigned to the class in __init__().

        :param verbose: Whether to display results to stdout
                while the Seldonian algorithms are running in each trial
        :type verbose: bool, defaults to False
        """
        from seldonian.RL.RL_runner import run_trial

        dataset = self.spec.dataset

        if self.datagen_method == "generate_episodes":
            # generate full-size datasets for each trial so that
            # we can reference them for each data_frac
            save_dir = os.path.join(self.results_dir, "regenerated_datasets")
            os.makedirs(save_dir, exist_ok=True)
            generate_behavior_policy_episodes(self.hyperparameter_and_setting_dict,self.n_trials,save_dir)

        run_seldonian_kwargs = dict(
            spec=self.spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
            datagen_method=self.datagen_method,
            hyperparameter_and_setting_dict=self.hyperparameter_and_setting_dict,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            batch_epoch_dict=self.batch_epoch_dict,
            verbose=verbose,
        )

        # # ## Run experiment
        sd_exp = SeldonianExperiment(model_name="qsa", results_dir=self.results_dir)

        sd_exp.run_experiment(**run_seldonian_kwargs)

    def run_baseline_experiment(self, baseline_model, verbose=False):
        """Run a supervised Seldonian experiment using the spec attribute
        assigned to the class in __init__().

        :param verbose: Whether to display results to stdout
                while the Seldonian algorithms are running in each trial
        :type verbose: bool, defaults to False
        """

        dataset = self.spec.dataset

        if self.datagen_method == "generate_episodes":
            # Generate n_trials resampled datasets of full length
            # These will be cropped to data_frac fractional size
            if verbose: print("checking for regenerated episodes")
            save_dir = os.path.join(self.results_dir, "regenerated_datasets")
            generate_behavior_policy_episodes(self.hyperparameter_and_setting_dict,self.n_trials,save_dir)
            if verbose:  print("Done checking for regenerated episodes\n")
        else:
            raise NotImplementedError(
                f"datagen_method {datagen_method} not supported for RL experiments")

        run_baseline_kwargs = dict(
            spec=self.spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
            datagen_method=self.datagen_method,
            hyperparameter_and_setting_dict=self.hyperparameter_and_setting_dict,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            batch_epoch_dict=self.batch_epoch_dict,
            verbose=verbose,
        )

        ## Run experiment
        bl_exp = BaselineExperiment(baseline_model=baseline_model, results_dir=self.results_dir)

        bl_exp.run_experiment(**run_baseline_kwargs)
        return

    def plot_importance_weights(
        self,
        n_trials,
        data_fracs,
        fontsize=12,
        title_fontsize=12,
        marker_size=20,
        save_format="pdf",
        show_title=True,
        custom_title=None,
        savename=None,
    ):
        """Plot the mean importance weights (over episodes) for 
        all trials in an experiment. Only uses the qsa_results/
        folder since that is the only experiment that is relevant.

        :param fontsize: The font size to use for the axis labels
        :type fontsize: int
        :param marker_size: The size of the points in each plots
        :type marker_size: float
        :param save_format: The file type for the saved plot
        :type save_format: str, defaults to "pdf"
        :param show_title: Whether to show the title at the top of the figure
        :type show_title: bool
        :param custom_title: A custom title 
        :type custom_title: str, defaults to None
        :param savename: If not None, the filename to which the figure
                will be saved on disk.
        :type savename: str, defaults to None
        """
        plt.style.use("bmh")
        regime = self.spec.dataset.regime
        if regime != "reinforcement_learning":
            raise ValueError(
                "Importance weights can only be plotted for reinforcement learning problems"
            )

        tot_data_size = self.hyperparameter_and_setting_dict['num_episodes']

        # Figure out what experiments we have from subfolders in results_dir
        is_parent_dir = os.path.join(self.results_dir,"qsa_results","importance_weights")
        cs_weights_dir = os.path.join(is_parent_dir,"candidate_selection")
        st_weights_dir = os.path.join(is_parent_dir,"safety_test")

        ## PLOTTING SETUP
        
        figsize = (12, 6)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
        locmin = matplotlib.ticker.LogLocator(
            base=10.0,
            subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            numticks=12,
        )

        ax.minorticks_on()
        ax.xaxis.set_major_locator(locmaj)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xscale("log")
        # SELDONIAN RESULTS SETUP
        for branch in ["cs","st"]:
            if branch == "cs":
                color="blue"
                label="Candidate selection"
                weights_dir = cs_weights_dir
            else: 
                color="red"
                label="Safety test"
                weights_dir = st_weights_dir
            
            IS_weights_dict = {
                data_frac:[] for data_frac in data_fracs
            } # keys: data_fracs, values: lists of mean IS weights for all trials at that data_frac
            for data_frac in data_fracs:
                for trial_i in range(n_trials):
                    filename = os.path.join(
                        weights_dir, f"importance_weights_frac_{data_frac:.4f}_trial_{trial_i}.pkl"
                    )
                    importance_weights = load_pickle(filename)
                    if importance_weights is not None:
                        mean_IS = np.mean(importance_weights)
                        IS_weights_dict[data_frac].append(mean_IS)

            good_data_fracs = np.array([key for key in IS_weights_dict if len(IS_weights_dict[key])>1])
            n_trials_good_list = np.array([len(IS_weights_dict[key]) for key in good_data_fracs])
            mean_good_IS_weights = np.array([np.mean(IS_weights_dict[key]) for key in good_data_fracs])
            ste_good_IS_weights = np.array([np.std(IS_weights_dict[good_data_fracs[ii]])/n_trials_good_list[ii] for ii in range(len(good_data_fracs))])
            
            ax.scatter(
                good_data_fracs*tot_data_size,
                mean_good_IS_weights,
                s=marker_size,
                color=color,
                marker="o",
                zorder=10,
                label=label)
            
            ax.fill_between(
                good_data_fracs*tot_data_size,
                mean_good_IS_weights - ste_good_IS_weights,
                mean_good_IS_weights + ste_good_IS_weights,
                color=color,
                alpha=0.5,
                zorder=10
            )
        
        ax.set_xlim(5,tot_data_size)
        ax.set_xlabel("Amount of data", fontsize=fontsize)
        ax.set_ylabel("Mean importance weight",fontsize=fontsize)
        ax.legend()

        if custom_title:
            title = custom_title
        else:
            title = f"Importance weights vs. amount of data"
        ax.set_title(title, y=1.05, fontsize=title_fontsize)

        if savename:
            plt.savefig(savename, format=save_format,bbox_inches="tight")
            print(f"Saved {savename}")
        else:
            plt.show()
