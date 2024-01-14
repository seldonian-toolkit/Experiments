""" Utilities used in the rest of the library """

import os, copy, pickle, math
import numpy as np

from seldonian.RL.RL_runner import (
    run_trial,
    create_agent_fromdict,
    run_trial_given_agent_and_env,
)
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.dataset import SupervisedDataSet, RLDataSet, CustomDataSet
from seldonian.utils.io_utils import load_pickle, save_pickle


def generate_behavior_policy_episodes(
    hyperparameter_and_setting_dict, n_trials, save_dir, verbose=False
):
    """Utility function for reinforcement learning to generate new episodes
    using the behavior policy to use in each trial.

    :param hyperparameter_and_setting_dict: Contains the number of episodes to generate,
        environment, agent, etc. needed for generating new episodes.
    :type hyperparameter_and_setting_dict: dict

    :param n_trials: The number of experiment trials to run per data fraction
    :param save_dir: The parent directory in which to save the
            regenerated_episodes
    :type save_dir: str
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        n_workers_for_episode_generation = hyperparameter_and_setting_dict[
            "n_workers_for_episode_generation"
        ]
    except:
        n_workers_for_episode_generation = 1

    if verbose:
        print("generating new episodes for each trial")
    for trial_i in range(n_trials):
        if verbose:
            print(f"Trial: {trial_i+1}/{self.n_trials}")
        savename = os.path.join(save_dir, f"regenerated_data_trial{trial_i}.pkl")
        if not os.path.exists(savename):
            if n_workers_for_episode_generation > 1:
                episodes = run_trial(
                    hyperparameter_and_setting_dict,
                    parallel=True,
                    n_workers=n_workers_for_episode_generation,
                )
            else:
                episodes = run_trial(hyperparameter_and_setting_dict, parallel=False)
            # Save episodes
            save_pickle(savename, episodes, verbose=verbose)
        else:
            if verbose:
                print(f"{savename} already created")
    return


def load_resampled_datasets(spec, results_dir, trial_i, data_frac, verbose=False):
    """Utility function for supervised learning to generate the
    resampled datasets to use in each trial. Resamples (with replacement)
    features, labels and sensitive attributes to create n_trials versions of these
    of the same shape as the inputs

    :param spec: A seldonian.spec.Spec object.
    :param results_dir: The directory in which results are saved for this trial
    :type results_dir: str
    :param trial_i: Trial index
    :type trial_i: int
    :param data_frac: data fraction
    :type data_frac: float

    :param verbose: boolean verbosity flag
    """

    # Primary dataset
    resampled_base_dir = os.path.join(results_dir, "resampled_datasets")
    # Check if forced candidate/safety data

    if spec.candidate_dataset is not None:
        resampled_cand_filename = os.path.join(
            resampled_base_dir, f"trial_{trial_i}_candidate_dataset.pkl"
        )
        resampled_cand_dataset = load_pickle(resampled_cand_filename)
        resampled_safety_filename = os.path.join(
            resampled_base_dir, f"trial_{trial_i}_safety_dataset.pkl"
        )
        resampled_safety_dataset = load_pickle(resampled_safety_filename)
        resampled_datasets = {
            "candidate_dataset": resampled_cand_dataset,
            "safety_dataset": resampled_safety_dataset,
        }
        num_datapoints_cand = int(
            round(data_frac * resampled_cand_dataset.num_datapoints)
        )
        num_datapoints_safety = int(
            round(data_frac * resampled_safety_dataset.num_datapoints)
        )

        n_points_dict = {
            "candidate_dataset": num_datapoints_cand,
            "safety_dataset": num_datapoints_safety,
        }
    else:
        resampled_filename = os.path.join(resampled_base_dir, f"trial_{trial_i}.pkl")
        resampled_dataset = load_pickle(resampled_filename)
        resampled_datasets = {"dataset": resampled_dataset}
        num_datapoints_tot = resampled_dataset.num_datapoints
        n_points = int(round(data_frac * num_datapoints_tot))
        n_points_dict = {"dataset": n_points}

    for key, val in n_points_dict.items():
        if val < 1:
            raise ValueError(
                f"This data_frac={data_frac} "
                f"results in {n_points} data points. "
                "Must have at least 1 data point to run a trial."
            )

    if spec.additional_datasets:
        addl_resampled_filename = os.path.join(
            results_dir, "resampled_datasets", f"trial_{trial_i}_addl_datasets.pkl"
        )
        additional_datasets = load_pickle(addl_resampled_filename)
    else:
        additional_datasets = {}

    return resampled_datasets, n_points_dict, additional_datasets


def load_regenerated_episodes(
    results_dir, trial_i, data_frac, orig_meta, verbose=False
):
    """Load the episodes generatd for each experiment trial.
    :param results_dir: The directory in which results are saved for this trial
    :type results_dir: str
    :param trial_i: Trial index
    :type trial_i: int
    :param data_frac: data fraction
    :type data_frac: float
    :param orig_meta: MetaData object from the original spec.dataset for this experiment
    
    """
    save_dir = os.path.join(results_dir, "regenerated_datasets")
    savename = os.path.join(save_dir, f"regenerated_data_trial{trial_i}.pkl")

    episodes_all = load_pickle(savename)
    # Take data_frac episodes from this df
    n_episodes_all = len(episodes_all)

    n_episodes_for_exp = int(round(n_episodes_all * data_frac))
    if n_episodes_for_exp < 1:
        raise ValueError(
            f"This data_frac={data_frac} "
            f"results in {n_episodes_for_exp} episodes. "
            "Must have at least 1 episode to run a trial."
        )

    if verbose:
        print(f"Orig dataset should have {n_episodes_all} episodes")
    if verbose:
        print(
            f"This dataset with data_frac={data_frac} should have"
            f" {n_episodes_for_exp} episodes"
        )

    # Take first n_episodes episodes
    episodes_for_exp = episodes_all[0:n_episodes_for_exp]
    assert len(episodes_for_exp) == n_episodes_for_exp

    dataset_for_exp = RLDataSet(
        episodes=episodes_for_exp,
        meta=orig_meta,
    )
    return dataset_for_exp


def prep_feat_labels(trial_dataset, n_points, include_sensitive_attrs=False):
    """Utility function for preparing features and labels
    for a given trial with n_points (given data frac)

    :param trial_dataset: The Seldonian dataset object containing trial data
    :param n_points: Number of points in this trial
    :type n_points: int
    :param include_sensitive_attrs: Whether to prep and return sensitive attributes
        as well.
    :type include_sensitive_attrs: bool
    """
    features = trial_dataset.features
    labels = trial_dataset.labels
    # Only use first n_points for this trial
    if type(features) == list:
        features = [x[:n_points] for x in features]
    else:
        features = features[:n_points]
    labels = labels[:n_points]

    if include_sensitive_attrs:
        sensitive_attrs = trial_dataset.sensitive_attrs
        sensitive_attrs = sensitive_attrs[:n_points]
        return features, labels, sensitive_attrs

    return features, labels


def prep_feat_labels_for_baseline(
    spec, results_dir, trial_i, data_frac, datagen_method, verbose
):
    """Utility function for preparing features and labels
    for a given baseline trial.

    :param spec: A seldonian.spec.Spec object.
    :param results_dir: The directory in which results are saved for this trial
    :type results_dir: str
    :param trial_i: Trial index
    :type trial_i: int
    :param data_frac: data fraction
    :type data_frac: float
    :param datagen_method: Method for generating the trial datasets.
    """
    if datagen_method == "resample":
        trial_datasets, n_points_dict, trial_addl_datasets = load_resampled_datasets(
            spec, results_dir, trial_i, data_frac, verbose=verbose
        )
        # If there are separate resampled candidate and safety datasets,
        # we merge them into a single dataset and then take the first n_points points
        # from the merged dataset
        if "candidate_dataset" in trial_datasets:
            trial_cand_dataset = trial_datasets["candidate_dataset"]
            trial_safety_dataset = trial_datasets["safety_dataset"]
            cand_features = trial_cand_dataset.features
            cand_labels = trial_cand_dataset.labels
            cand_sensitive_attrs = trial_cand_dataset.sensitive_attrs
            safety_features = trial_safety_dataset.features
            safety_labels = trial_safety_dataset.labels
            safety_sensitive_attrs = trial_safety_dataset.sensitive_attrs
            merged_features = np.vstack([cand_features, safety_features])
            merged_labels = np.hstack([cand_labels, safety_labels])
            merged_sensitive_attrs = np.vstack(
                [cand_sensitive_attrs, safety_sensitive_attrs]
            )

            merged_trial_dataset = SupervisedDataSet(
                features=merged_features,
                labels=merged_labels,
                sensitive_attrs=merged_sensitive_attrs,
                num_datapoints=len(merged_labels),
                meta=trial_cand_dataset.meta,
            )
            n_points_cand = n_points_dict["candidate_dataset"]
            n_points_safety = n_points_dict["safety_dataset"]
            n_points_merged = n_points_cand + n_points_safety
            features, labels = prep_feat_labels(merged_trial_dataset, n_points_merged)
        else:
            trial_dataset = trial_datasets["dataset"]
            n_points = n_points_dict["dataset"]
            features, labels = prep_feat_labels(trial_dataset, n_points)

    else:
        raise NotImplementedError(
            f"datagen_method: {datagen_method} " f"not supported for regime: {regime}"
        )

    return features, labels


def prep_data_for_fairlearn(
    spec,
    results_dir,
    trial_i,
    data_frac,
    datagen_method,
    fairlearn_sensitive_feature_names,
    verbose,
):
    """Utility function for preparing features and labels
    for a given fairlearn trial.

    :param spec: A seldonian.spec.Spec object.
    :param results_dir: The directory in which results are saved for this trial
    :type results_dir: str
    :param trial_i: Trial index
    :type trial_i: int
    :param data_frac: data fraction
    :type data_frac: float
    :param datagen_method: Method for generating the trial datasets.
    :param fairlearn_sensitive_feature_names: List of names of the sensitive attributes
        that fairlearn will use.
    """
    if datagen_method == "resample":
        trial_datasets, n_points_dict, trial_addl_datasets = load_resampled_datasets(
            spec, results_dir, trial_i, data_frac, verbose=verbose
        )
        # If there are separate resampled candidate and safety datasets,
        # we merge them into a single dataset and then take the first n_points points
        # from the merged dataset
        if "candidate_dataset" in trial_datasets:
            trial_cand_dataset = trial_datasets["candidate_dataset"]
            trial_safety_dataset = trial_datasets["safety_dataset"]
            cand_features = trial_cand_dataset.features
            cand_labels = trial_cand_dataset.labels
            cand_sensitive_attrs = trial_cand_dataset.sensitive_attrs
            safety_features = trial_safety_dataset.features
            safety_labels = trial_safety_dataset.labels
            safety_sensitive_attrs = trial_safety_dataset.sensitive_attrs
            merged_features = np.vstack([cand_features, safety_features])
            merged_labels = np.hstack([cand_labels, safety_labels])
            merged_sensitive_attrs = np.vstack(
                [cand_sensitive_attrs, safety_sensitive_attrs]
            )

            merged_trial_dataset = SupervisedDataSet(
                features=merged_features,
                labels=merged_labels,
                sensitive_attrs=merged_sensitive_attrs,
                num_datapoints=len(merged_labels),
                meta=trial_cand_dataset.meta,
            )
            n_points_cand = n_points_dict["candidate_dataset"]
            n_points_safety = n_points_dict["safety_dataset"]
            n_points_merged = n_points_cand + n_points_safety
            features, labels, sensitive_attrs = prep_feat_labels(
                merged_trial_dataset, n_points_merged, include_sensitive_attrs=True
            )
            sensitive_col_indices = [
                merged_trial_dataset.sensitive_col_names.index(col)
                for col in fairlearn_sensitive_feature_names
            ]

            fairlearn_sensitive_features = np.squeeze(
                sensitive_attrs[:, sensitive_col_indices]
            )
        else:
            trial_dataset = trial_datasets["dataset"]
            n_points = n_points_dict["dataset"]
            features, labels = prep_feat_labels(trial_dataset, n_points)
            sensitive_col_indices = [
                trial_dataset.sensitive_col_names.index(col)
                for col in fairlearn_sensitive_feature_names
            ]

            fairlearn_sensitive_features = np.squeeze(
                trial_dataset.sensitive_attrs[:, sensitive_col_indices]
            )[:n_points]

    else:
        raise NotImplementedError(
            f"datagen_method: {datagen_method} " f"not supported for regime: {regime}"
        )

    return features, labels, fairlearn_sensitive_features


def prep_custom_data(trial_dataset, n_points, include_sensitive_attrs=False):
    """Utility function for preparing data and sensitive attributes
    for the custom regime for a given trial with n_points (given data frac)

    :param trial_dataset: The Seldonian dataset object containing trial data
    :param n_points: Number of points in this trial
    :type n_points: int
    :param include_sensitive_attrs: Whether to prep and return sensitive attributes
        as well.
    :type include_sensitive_attrs: bool
    """
    # Only use first n_points for this trial
    data = trial_dataset.data[:n_points]

    if include_sensitive_attrs:
        sensitive_attrs = trial_dataset.sensitive_attrs[:n_points]
        return data, sensitive_attrs

    return data


def setup_SA_spec_for_exp(
    spec,
    regime,
    results_dir,
    trial_i,
    data_frac,
    datagen_method,
    batch_epoch_dict,
    kwargs,
    perf_eval_kwargs,
):
    """Utility function for setting up the spec object
    to use for a Seldonian algorithm trial

    :param spec: A seldonian.spec.Spec object.
    :param regime: The category of ML problem. 
    :param results_dir: The directory in which results are saved for this trial
    :type results_dir: str
    :param trial_i: Trial index for this data fraction
    :type trial_i: int
    :param data_frac: data fraction
    :type data_frac: float
    :param datagen_method: Method for generating the trial datasets.
    :param batch_epoch_dict: A dictionary where keys are data fractions
        and values are [batch_size,num_epochs]

    :return: spec_for_exp, the spec object ready for running this Seldonian trial.
    """
    if regime == "supervised_learning":
        if datagen_method == "resample":
            (
                trial_datasets,
                n_points_dict,
                trial_addl_datasets,
            ) = load_resampled_datasets(spec, results_dir, trial_i, data_frac)

        else:
            raise NotImplementedError(
                f"Eval method {datagen_method} " f"not supported for regime={regime}"
            )

        # Make a new spec object which we will modify
        spec_for_exp = copy.deepcopy(spec)

        # Check if we have forced candidate/safety datasets
        if "candidate_dataset" in trial_datasets:
            trial_cand_dataset = trial_datasets["candidate_dataset"]
            n_points_cand = n_points_dict["candidate_dataset"]
            trial_safety_dataset = trial_datasets["safety_dataset"]
            n_points_safety = n_points_dict["safety_dataset"]

            cand_features, cand_labels, cand_sensitive_attrs = prep_feat_labels(
                trial_cand_dataset, n_points_cand, include_sensitive_attrs=True
            )
            safety_features, safety_labels, safety_sensitive_attrs = prep_feat_labels(
                trial_safety_dataset, n_points_safety, include_sensitive_attrs=True
            )

            cand_dataset_for_exp = SupervisedDataSet(
                features=cand_features,
                labels=cand_labels,
                sensitive_attrs=cand_sensitive_attrs,
                num_datapoints=n_points_cand,
                meta=trial_cand_dataset.meta,
            )

            safety_dataset_for_exp = SupervisedDataSet(
                features=safety_features,
                labels=safety_labels,
                sensitive_attrs=safety_sensitive_attrs,
                num_datapoints=n_points_safety,
                meta=trial_safety_dataset.meta,
            )

            # Update candidate and safety datasets
            spec_for_exp.candidate_dataset = cand_dataset_for_exp
            spec_for_exp.safety_dataset = safety_dataset_for_exp
        else:
            trial_dataset = trial_datasets["dataset"]
            n_points = n_points_dict["dataset"]
            features, labels, sensitive_attrs = prep_feat_labels(
                trial_dataset, n_points, include_sensitive_attrs=True
            )

            dataset_for_exp = SupervisedDataSet(
                features=features,
                labels=labels,
                sensitive_attrs=sensitive_attrs,
                num_datapoints=n_points,
                meta=trial_dataset.meta,
            )

            # Update primary dataset
            spec_for_exp.dataset = dataset_for_exp

        # Check for addl datasets
        if trial_addl_datasets != {}:
            additional_datasets_for_exp = {}
            for constraint_str in trial_addl_datasets:
                additional_datasets_for_exp[constraint_str] = {}
                for bn in trial_addl_datasets[constraint_str]:
                    additional_datasets_for_exp[constraint_str][bn] = {}
                    this_dict = trial_addl_datasets[constraint_str][bn]
                    addl_batch_size = this_dict.get("batch_size")
                    if addl_batch_size:
                        additional_datasets_for_exp[constraint_str][bn][
                            "batch_size"
                        ] = addl_batch_size

                    # Check to see if candidate/safety datasets are explicitly provided
                    if "candidate_dataset" in this_dict:
                        keys = ["candidate_dataset", "safety_dataset"]
                    else:
                        keys = ["dataset"]

                    for key in keys:
                        addl_trial_dataset = this_dict[key]
                        num_datapoints_addl = addl_trial_dataset.num_datapoints
                        addl_n_points = int(round(data_frac * num_datapoints_addl))

                        (
                            addl_features,
                            addl_labels,
                            addl_sensitive_attrs,
                        ) = prep_feat_labels(
                            addl_trial_dataset,
                            addl_n_points,
                            include_sensitive_attrs=True,
                        )

                        addl_dataset_for_exp = SupervisedDataSet(
                            features=addl_features,
                            labels=addl_labels,
                            sensitive_attrs=addl_sensitive_attrs,
                            num_datapoints=addl_n_points,
                            meta=addl_trial_dataset.meta,
                        )
                        additional_datasets_for_exp[constraint_str][bn][
                            key
                        ] = addl_dataset_for_exp

            spec_for_exp.additional_datasets = additional_datasets_for_exp

    elif regime == "reinforcement_learning":
        hyperparameter_and_setting_dict = kwargs["hyperparameter_and_setting_dict"]

        if datagen_method == "generate_episodes":
            # Sample from resampled dataset on disk of n_episodes
            dataset_for_exp = load_regenerated_episodes(
                results_dir, trial_i, data_frac, spec.dataset.meta
            )

            # Make a new spec object from a copy of spec, where the
            # only thing that is different is the dataset

            spec_for_exp = copy.deepcopy(spec)
            spec_for_exp.dataset = dataset_for_exp
        else:
            raise NotImplementedError(
                f"Eval method {datagen_method} not supported for regime={regime}"
            )

    elif regime == "custom":
        if datagen_method == "resample":
            (
                trial_datasets,
                n_points_dict,
                trial_addl_datasets,
            ) = load_resampled_datasets(spec, results_dir, trial_i, data_frac)

        else:
            raise NotImplementedError(
                f"Eval method {datagen_method} " f"not supported for regime={regime}"
            )

        # Make a new spec object which we will modify
        spec_for_exp = copy.deepcopy(spec)

        # Check if we have forced candidate/safety datasets
        if "candidate_dataset" in trial_datasets:
            trial_cand_dataset = trial_datasets["candidate_dataset"]
            n_points_cand = n_points_dict["candidate_dataset"]
            trial_safety_dataset = trial_datasets["safety_dataset"]
            n_points_safety = n_points_dict["safety_dataset"]

            cand_data, cand_sensitive_attrs = prep_custom_data(
                trial_cand_dataset, n_points_cand, include_sensitive_attrs=True
            )
            safety_data, safety_sensitive_attrs = prep_custom_data(
                trial_safety_dataset, n_points_safety, include_sensitive_attrs=True
            )

            cand_dataset_for_exp = CustomDataSet(
                data=cand_data,
                sensitive_attrs=cand_sensitive_attrs,
                num_datapoints=n_points_cand,
                meta=trial_cand_dataset.meta,
            )

            safety_dataset_for_exp = CustomDataSet(
                data=safety_data,
                sensitive_attrs=safety_sensitive_attrs,
                num_datapoints=n_points_safety,
                meta=trial_safety_dataset.meta,
            )

            # Update candidate and safety datasets
            spec_for_exp.candidate_dataset = cand_dataset_for_exp
            spec_for_exp.safety_dataset = safety_dataset_for_exp
        else:
            trial_dataset = trial_datasets["dataset"]
            n_points = n_points_dict["dataset"]
            data, sensitive_attrs = prep_custom_data(
                trial_dataset, n_points, include_sensitive_attrs=True
            )

            dataset_for_exp = CustomDataSet(
                data=data,
                sensitive_attrs=sensitive_attrs,
                num_datapoints=n_points,
                meta=trial_dataset.meta,
            )

            # Update primary dataset
            spec_for_exp.dataset = dataset_for_exp

        # Check for addl datasets
        if trial_addl_datasets != {}:
            additional_datasets_for_exp = {}
            for constraint_str in trial_addl_datasets:
                additional_datasets_for_exp[constraint_str] = {}
                for bn in trial_addl_datasets[constraint_str]:
                    additional_datasets_for_exp[constraint_str][bn] = {}
                    this_dict = trial_addl_datasets[constraint_str][bn]
                    addl_batch_size = this_dict.get("batch_size")
                    if addl_batch_size:
                        additional_datasets_for_exp[constraint_str][bn][
                            "batch_size"
                        ] = addl_batch_size

                    # Check to see if candidate/safety datasets are explicitly provided
                    if "candidate_dataset" in this_dict:
                        keys = ["candidate_dataset", "safety_dataset"]
                    else:
                        keys = ["dataset"]

                    for key in keys:
                        addl_trial_dataset = this_dict[key]
                        num_datapoints_addl = addl_trial_dataset.num_datapoints
                        addl_n_points = int(round(data_frac * num_datapoints_addl))

                        (
                            addl_data,
                            addl_sensitive_attrs,
                        ) = prep_custom_data(
                            addl_trial_dataset,
                            addl_n_points,
                            include_sensitive_attrs=True,
                        )

                        addl_dataset_for_exp = CustomDataSet(
                            data=addl_data,
                            sensitive_attrs=addl_sensitive_attrs,
                            num_datapoints=addl_n_points,
                            meta=addl_trial_dataset.meta,
                        )
                        additional_datasets_for_exp[constraint_str][bn][
                            key
                        ] = addl_dataset_for_exp

            spec_for_exp.additional_datasets = additional_datasets_for_exp
    """ If optimizing using gradient descent and using mini-batches,
     update the batch_size and n_epochs using batch_epoch_dict """
    if (
        batch_epoch_dict != {}
        and spec_for_exp.optimization_technique == "gradient_descent"
        and (spec_for_exp.optimization_hyperparams["use_batches"] == True)
    ):
        batch_size, n_epochs = batch_epoch_dict[data_frac]
        spec_for_exp.optimization_hyperparams["batch_size"] = batch_size
        spec_for_exp.optimization_hyperparams["n_epochs"] = n_epochs

    return spec_for_exp


def generate_episodes_and_calc_J(**kwargs):
    """Calculate the expected discounted return
    by generating episodes

    :return: (episodes, J), where episodes is the list
            of generated ground truth episodes and J is
            the expected discounted return
    :rtype: (List(Episode),float)
    """
    # Get trained model weights from running the Seldonian algo
    model = kwargs["model"]
    new_params = model.policy.get_params()

    # create env and agent
    hyperparameter_and_setting_dict = kwargs["hyperparameter_and_setting_dict"]
    agent = create_agent_fromdict(hyperparameter_and_setting_dict)
    env = hyperparameter_and_setting_dict["env"]

    # set agent's weights to the trained model weights
    agent.set_new_params(new_params)

    # generate episodes
    num_episodes = kwargs["n_episodes_for_eval"]
    episodes = run_trial_given_agent_and_env(
        agent=agent, env=env, num_episodes=num_episodes
    )

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards, env.gamma) for ep in episodes])
    J = np.mean(returns)
    return episodes, J


def batch_predictions(model, solution, X_test, **kwargs):
    """Run model forward pass in batches.

    :param model: A model object with a .predict(theta,X) method
    :param solution: Model weights to set before making the forward pass
    :param X_test: The features to batch up

    :return: y_pred, the combined predictions in a flattened array
    """
    batch_size = kwargs["eval_batch_size"]
    
    if type(X_test) == list:
        N_eval = len(X_test[0])
    else:
        N_eval = len(X_test)
    
    if "N_output_classes" in kwargs:
        N_output_classes = kwargs["N_output_classes"]
        y_pred = np.zeros((N_eval, N_output_classes))
    else:
        y_pred = np.zeros(N_eval)
    
    num_batches = math.ceil(N_eval / batch_size)
    batch_start = 0
    
    for i in range(num_batches):
        batch_end = batch_start + batch_size

        if type(X_test) == list:
            X_test_batch = [x[batch_start:batch_end] for x in X_test]
        else:
            X_test_batch = X_test[batch_start:batch_end]
        y_pred[batch_start:batch_end] = model.predict(solution, X_test_batch)
        batch_start = batch_end
    return y_pred


def batch_predictions_custom_regime(model, solution, test_data, **kwargs):
    """Run model forward pass in batches for the custom regime.

    :param model: A model object with a .predict(theta,data) method
    :param solution: Model weights to set before making the forward pass
    :param test_data: The input data to the model to batch up

    :return: y_pred, the combined predictions in a flattened array
    """
    batch_size = kwargs["eval_batch_size"]
    N_eval = len(test_data)
    if "N_output_classes" in kwargs:
        N_output_classes = kwargs["N_output_classes"]
        y_pred = np.zeros((N_eval, N_output_classes))
    else:
        y_pred = np.zeros(N_eval)
    num_batches = math.ceil(N_eval / batch_size)
    batch_start = 0
    for i in range(num_batches):
        batch_end = batch_start + batch_size
        test_data_batch = test_data[batch_start:batch_end]
        y_pred[batch_start:batch_end] = model.predict(solution, test_data_batch)
        batch_start = batch_end
    return y_pred


def make_batch_epoch_dict_fixedniter(niter, data_fracs, N_max, batch_size):
    """
    Convenience function for figuring out the number of epochs necessary
    to ensure that at each data fraction, the total
    number of iterations (and batch size) will stay fixed.

    :param niter: The total number of iterations you want run at every data_frac
    :type niter: int
    :param data_fracs: 1-D array of data fractions
    :type data_fracs: np.ndarray
    :param N_max: The maximum number of data points in the optimization process
    :type N_max: int
    :param batch_size: The fixed batch size
    :type batch_size: int

    :return batch_epoch_dict: A dictionary where keys are data fractions
        and values are [batch_size,num_epochs]
    """
    data_sizes = (
        data_fracs * N_max
    )  # number of points used in candidate selection in each data frac
    n_batches = data_sizes / batch_size  # number of batches in each data frac
    n_batches = np.array([math.ceil(x) for x in n_batches])
    n_epochs_arr = (
        niter / n_batches
    )  # number of epochs needed to get to at least niter iterations in each data frac
    n_epochs_arr = np.array([math.ceil(x) for x in n_epochs_arr])
    batch_epoch_dict = {
        data_fracs[ii]: [batch_size, n_epochs_arr[ii]] for ii in range(len(data_fracs))
    }
    return batch_epoch_dict


def make_batch_epoch_dict_min_sample_repeat(
    niter_min, data_fracs, N_max, batch_size, num_repeats
):
    """
    Convenience function for figuring out the number of epochs necessary
    to ensure that the number of iterations for each data frac is:
    max(niter_min,# of iterations such that each sample is seen num_repeat times)

    :param niter_min: The minimum total number of iterations you want run at every data_frac
    :type niter_min: int
    :param data_fracs: 1-D array of data fractions
    :type data_fracs: np.ndarray
    :param N_max: The maximum number of data points in the optimization process
    :type N_max: int
    :param batch_size: The fixed batch size
    :type batch_size: int
    :param num_repeats: The minimum number of times each sample must be seen in the optimization process
    :type num_repeats: int

    :return batch_epoch_dict: A dictionary where keys are data fractions
        and values are [batch_size,num_epochs]
    """
    batch_epoch_dict = {}
    n_epochs_arr = np.zeros_like(data_fracs)
    for data_frac in data_fracs:
        niter2 = num_repeats * N_max * data_frac / batch_size
        if niter2 > niter_min:
            num_epochs = num_repeats
        else:
            n_batches = max(1, N_max * data_frac / batch_size)
            num_epochs = math.ceil(niter_min / n_batches)
        batch_epoch_dict[data_frac] = [batch_size, num_epochs]

    return batch_epoch_dict


def has_failed(g):
    """Condition for whether a value of g is unsafe. This is used
    to determine the failure rate in the right-most plot of the experiments plots.

    :param g: The expected value of a single behavioral constraint function
    :type g: float

    :return: True if g is unsafe, False if g is safe
    """
    return g > 0 or np.isnan(g)


def trial_arg_chunker(data_fracs, n_trials, n_workers):
    """
    Convenience function for parallel processing that chunks up 
    the data fractions and trial indices as arguments
    for use in the map function of a ProcessPoolExecutor.

    :param data_fracs: 1-D array of data fractions
    :type data_fracs: np.ndarray
    :param n_trials: The number of trials per data fraction
    :type n_trials: int
    :param n_workers: The number of parallel processes that will be used.
    :type n_workers: int
    

    :return chunked_list: A list of lists of tuples (data_frac,trial_index)
    """
    n_tot = len(data_fracs) * n_trials
    chunk_size = n_tot // n_workers
    chunk_sizes = []
    for i in range(0, n_tot, chunk_size):
        if (i + chunk_size) > n_tot:
            chunk_sizes.append(n_tot - i)
        else:
            chunk_sizes.append(chunk_size)
    assert sum(chunk_sizes) == n_tot
    # flatten data fracs and trials so we can make chunked tuples ((data_frac,trial_index),...)
    data_fracs_vector = np.array([x for x in data_fracs for y in range(n_trials)])
    trials_vector = np.array(
        [x for y in range(len(data_fracs)) for x in range(n_trials)]
    )
    chunked_list = []
    start_ix = 0
    for chunk_size in chunk_sizes:
        chunked_list.append(
            [
                [data_fracs_vector[ii], trials_vector[ii]]
                for ii in range(start_ix, start_ix + chunk_size)
            ]
        )
        start_ix += chunk_size
    return chunked_list


def supervised_initial_solution_fn(m, x, y):
    """A common initial solution function used in supervised learning.
    Just a wrapper for the model.fit() method.

    :param m: SeldonianModel instance
    :param x: features
    :param y: labels

    :return: The output of the fit() method,
        which are model weights as a flattened 1D array
    """
    return m.fit(x, y)
