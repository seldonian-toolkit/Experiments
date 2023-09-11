""" Module for Plotting Selection Related Plots"""

import os
import pickle
import autograd.numpy as np  # Thinly-wrapped version of Numpy
import pandas as pd
import matplotlib
import scipy
import matplotlib.pyplot as plt
from matplotlib import style

from seldonian.utils.io_utils import save_pickle, load_pickle
from seldonian.utils.stats_utils import tinv

seldonian_model_set = set(["qsa", "sa"])
plot_colormap = matplotlib.cm.get_cmap("tab10")


class SelectionPlots:
    def __init__(
        self,
        spec,
        n_trials,
        data_fracs,
        results_dir,
        n_bootstrap_trials,
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

        :param results_dir: The directory in which to save the results
        :type results_dir: str
        """
        self.spec = spec
        self.n_trials = n_trials
        self.data_fracs = data_fracs
        self.results_dir = results_dir
        self.n_bootstrap_trials = n_bootstrap_trials

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
            plt.xlabel("Dataset Size")
            plt.ylabel("Bootstrap Estimate of P(pass)")


    def make_trial_probpass_est_plot(
        self,
        all_safety_frac,
        bound_type,
        alpha
    ):
        """
        Creates plot of the estimated probability of passing for each future_safety_frac
            for the given safety_frac

        :param all_safety_frac: array of containing all values of safety frac being considered
        :type pass_count: np.array
        :param bound_type: indicates what time of confidence bound to put around
        :type bound_type: string
        :param alpha: confidence parameter
        :type num_bootstrap_samples: float
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
                                trial_probpass_est + bootstrap_trial_ste, alpha=alpha)

                    elif bound_type == "clopper-pearson":
                        trial_pass_count = np.nansum(all_bootstrap_trial_info[
                            safety_frac][future_safety_frac][trial_i], axis=1)
                        lower_bound, upper_bound = self.clopper_pearson_bound(
                                trial_pass_count, self.n_bootstrap_trials)
                        lower_bound = np.nan_to_num(lower_bound) # If lower bound nan, set to 0.
                        plt.fill_between(all_data_size, lower_bound, upper_bound, alpha=alpha)

                    elif bound_type == "ttest":
                        lower_bound, upper_bound = self.ttest_bound(all_bootstrap_trial_info[
                            safety_frac][future_safety_frac][trial_i])
                        plt.fill_between(all_data_size, lower_bound, upper_bound, alpha=alpha)

                plt.xscale("log")
                plt.title(f"rho {safety_frac:.2f}, trial {trial_i}")
                plt.legend(title="future safety rho")
                plt.xlabel("Dataset Size")
                plt.ylabel("Estimated Probability of Passing")


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
        plt.ylabel("Proportion of Data in Safety Dataset")
        plt.xlabel("Dataset Size")




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
