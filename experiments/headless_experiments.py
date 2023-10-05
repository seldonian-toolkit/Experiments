""" Module for running Seldonian Experiments """
import copy
import os
import numpy as np
from functools import partial

from .experiments import Experiment 
from . import headless_utils
from .experiment_utils import batch_predictions

from seldonian.dataset import SupervisedDataSet
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

import warnings
from seldonian.warnings.custom_warnings import *

warnings.filterwarnings("ignore", category=FutureWarning)


class HeadlessSeldonianExperiment(Experiment):
    def __init__(self, model_name, results_dir):
        """Class for running Seldonian experiments

        :param model_name: The string name of the Seldonian model,
                only option is currently: 'headless_qsa' (quasi-Seldonian algorithm)
        :type model_name: str

        :param results_dir: Parent directory for saving any
                experimental results
        :type results_dir: str

        """
        super().__init__(model_name, results_dir)
        if self.model_name != "headless_qsa":
            raise NotImplementedError(
                "Headless Seldonian experiments for model: "
                f"{self.model_name} are not supported."
            )

    def run_experiment(self, **kwargs):
        """Run the Seldonian experiment"""
        n_workers = kwargs["n_workers"]
        partial_kwargs = {
            key: kwargs[key] for key in kwargs if key not in ["data_fracs", "n_trials"]
        }
        # Pass partial_kwargs onto self.QSA()
        helper = partial(self.run_trial, **partial_kwargs)

        data_fracs = kwargs["data_fracs"]
        n_trials = kwargs["n_trials"]
        data_fracs_vector = np.array([x for x in data_fracs for y in range(n_trials)])
        trials_vector = np.array(
            [x for y in range(len(data_fracs)) for x in range(n_trials)]
        )

        if n_workers == 1:
            for ii in range(len(data_fracs_vector)):
                data_frac = data_fracs_vector[ii]
                trial_i = trials_vector[ii]
                helper(data_frac, trial_i)
        elif n_workers > 1:
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp.get_context("fork")
            ) as ex:
                results = tqdm(
                    ex.map(helper, data_fracs_vector, trials_vector),
                    total=len(data_fracs_vector),
                )
                for exc in results:
                    if exc:
                        print(exc)
        else:
            raise ValueError(f"n_workers value of {n_workers} must be >=1 ")

        self.aggregate_results(**kwargs)

    def run_trial(self, data_frac, trial_i, **kwargs):
        """Run a trial of the quasi-Seldonian algorithm. 
        First, obtain the latent features by training 
        a full model on the candidate data, then creating 
        the headless model, then passing all data
        through the headless model.

        :param data_frac: Fraction of overall dataset size to use
        :type data_frac: float

        :param trial_i: The index of the trial
        :type trial_i: int
        """
        spec = kwargs["spec"]
        verbose = kwargs["verbose"]
        datagen_method = kwargs["datagen_method"]
        perf_eval_fn = kwargs["perf_eval_fn"]
        perf_eval_kwargs = kwargs["perf_eval_kwargs"]
        constraint_eval_fns = kwargs["constraint_eval_fns"]
        constraint_eval_kwargs = kwargs["constraint_eval_kwargs"]
        batch_epoch_dict = kwargs["batch_epoch_dict"]

        # Headless kwargs
        full_pretraining_model=kwargs["full_pretraining_model"]
        initial_state_dict=kwargs["initial_state_dict"]
        headless_pretraining_model=kwargs["headless_pretraining_model"]
        head_layer_names=kwargs["head_layer_names"]
        latent_feature_shape=kwargs["latent_feature_shape"]

        batch_epoch_dict_pretraining=kwargs["batch_epoch_dict_pretraining"]
        candidate_batch_size_pretraining,num_epochs_pretraining = batch_epoch_dict_pretraining[data_frac]
        safety_batch_size_pretraining=kwargs["safety_batch_size_pretraining"]
        loss_func_pretraining=kwargs["loss_func_pretraining"]
        learning_rate_pretraining=kwargs["learning_rate_pretraining"]
        pretraining_device=kwargs["pretraining_device"]
        
        regime = spec.dataset.regime

        trial_dir = os.path.join(self.results_dir, f"{self.model_name}_results", "trial_data")

        savename = os.path.join(
            trial_dir, f"data_frac_{data_frac:.4f}_trial_{trial_i}.csv"
        )

        if os.path.exists(savename):
            if verbose:
                print(
                    f"Trial {trial_i} already run for "
                    f"this data_frac: {data_frac}. Skipping this trial. "
                )
            return

        os.makedirs(trial_dir, exist_ok=True)

        ##############################################
        """ Setup for running Seldonian algorithm """
        ##############################################

        if regime != "supervised_learning":
            raise NotImplementedError(
                "Headless experiments are only available for supervised_learning")

        if datagen_method != "resample":
            raise NotImplementedError(
                f"Eval method {datagen_method} "
                f"not supported for headless experiments. "
                f"Only 'resample' eval method is available."
            )

        if batch_epoch_dict == {} and spec.optimization_technique == "gradient_descent":
            warning_msg = (
                "WARNING: No batch_epoch_dict was provided. "
                "Each data_frac will use the same values "
                "for batch_size and n_epochs. "
                "This can have adverse effects if you use batches in gradient descent, "
                "especially for small values of data_frac."
            )
            warnings.warn(warning_msg)

        # Load resampled data in original feature format
        resampled_filename = os.path.join(
            self.results_dir, "resampled_dataframes", f"trial_{trial_i}.pkl"
        )
        resampled_dataset = load_pickle(resampled_filename)
        num_datapoints_tot = resampled_dataset.num_datapoints
        n_points = int(round(data_frac * num_datapoints_tot))

        if verbose:
            print(
                f"Using resampled dataset {resampled_filename} "
                f"with {num_datapoints_tot} datapoints"
            )
            if n_points < 1:
                raise ValueError(
                    f"This data_frac={data_frac} "
                    f"results in {n_points} data points. "
                    "Must have at least 1 data point to run a trial."
                )

        features = resampled_dataset.features
        labels = resampled_dataset.labels
        sensitive_attrs = resampled_dataset.sensitive_attrs
        # Only use first n_points for this trial
        if type(features) == list:
            raise ValueError("Features must be in arrays for headless experiments")

        features = features[:n_points]
        labels = labels[:n_points]
        sensitive_attrs = sensitive_attrs[:n_points]

        if verbose:
            print(f"With data_frac: {data_frac}, have {n_points} data points")

        # Obtain latent features by training the full model 
        # and then passing the data through a headless version of this model

        # First re-initialize weights
        full_pretraining_model.load_state_dict(initial_state_dict)
        
        latent_features = headless_utils.generate_latent_features(
            full_pretraining_model=full_pretraining_model,
            headless_pretraining_model=headless_pretraining_model,
            head_layer_names=head_layer_names,
            orig_features=features,
            labels=labels, 
            latent_feature_shape=latent_feature_shape,
            frac_data_in_safety=spec.frac_data_in_safety, 
            candidate_batch_size=candidate_batch_size_pretraining, 
            safety_batch_size=safety_batch_size_pretraining,
            loss_func=loss_func_pretraining,
            learning_rate=learning_rate_pretraining,
            num_epochs=num_epochs_pretraining,
            device=pretraining_device)
        
        dataset_for_experiment = SupervisedDataSet(
            features=latent_features,
            labels=labels,
            sensitive_attrs=sensitive_attrs,
            num_datapoints=n_points,
            meta=resampled_dataset.meta,
        )

        # Make a new spec object
        # and update the dataset

        spec_for_experiment = copy.deepcopy(spec)
        spec_for_experiment.dataset = dataset_for_experiment

        # If optimizing using gradient descent,
        # and using mini-batches,
        # update the batch_size and n_epochs
        # using batch_epoch_dict
        if spec_for_experiment.optimization_technique == "gradient_descent":
            if spec_for_experiment.optimization_hyperparams["use_batches"] == True:
                if verbose:
                    print("Using batches in Seldonian trial")
                batch_size, n_epochs = batch_epoch_dict[data_frac]
                spec_for_experiment.optimization_hyperparams["batch_size"] = batch_size
                spec_for_experiment.optimization_hyperparams["n_epochs"] = n_epochs
        ################################
        """" Run Seldonian algorithm """
        ################################

        try:
            SA = SeldonianAlgorithm(spec_for_experiment)
            passed_safety, solution = SA.run(write_cs_logfile=verbose, debug=verbose)

        except (ValueError, ZeroDivisionError):
            passed_safety = False
            solution = "NSF"

        if verbose:
            print("Solution from running seldonian algorithm:")
            print(solution)
            print()

        # Handle whether solution was found
        solution_found = True
        if type(solution) == str and solution == "NSF":
            solution_found = False

        #########################################################
        """" Calculate performance and safety on ground truth """
        #########################################################

        if solution_found:
            solution = copy.deepcopy(solution)
            # If passed the safety test, calculate performance
            # using solution
            if passed_safety:
                if verbose:
                    print("Passed safety test! Calculating performance")
                
                #############################
                """ Calculate performance """
                #############################
                if regime == "supervised_learning":
                    # First need to pass all images through the pretrained headless model
                    test_data_loaders = perf_eval_kwargs["test_data_loaders"]
                    X_test = headless_utils.forward_pass_all_features(
                        test_data_loaders,
                        headless_pretraining_model,
                        latent_feature_shape,
                        device=pretraining_device
                    )
                    # X_test = perf_eval_kwargs["X"]
                    Y_test = perf_eval_kwargs["y"]
                    model = SA.model
                    # Batch the prediction if specified
                    if "eval_batch_size" in perf_eval_kwargs:
                        y_pred = batch_predictions(
                            model=model,
                            solution=solution,
                            X_test=X_test,
                            **perf_eval_kwargs,
                        )
                    else:
                        y_pred = model.predict(solution, X_test)

                    performance = perf_eval_fn(y_pred, model=model, **perf_eval_kwargs)

                if verbose:
                    print(f"Performance = {performance}")

                ########################################
                """ Calculate safety on ground truth """
                ########################################
                if verbose:
                    print(
                        "Determining whether solution "
                        "is actually safe on ground truth"
                    )

                if constraint_eval_fns == []:
                    constraint_eval_kwargs["model"] = model
                    constraint_eval_kwargs["X_test"] = X_test
                    if "eval_batch_size" in perf_eval_kwargs:
                        constraint_eval_kwargs["eval_batch_size"] = perf_eval_kwargs["eval_batch_size"]
                    constraint_eval_kwargs["spec_orig"] = spec
                    constraint_eval_kwargs["spec_for_experiment"] = spec_for_experiment
                    constraint_eval_kwargs["regime"] = regime
                    constraint_eval_kwargs["branch"] = "safety_test"
                    constraint_eval_kwargs["verbose"] = verbose

                gvec = self.evaluate_constraint_functions(
                    solution=solution,
                    constraint_eval_fns=constraint_eval_fns,
                    constraint_eval_kwargs=constraint_eval_kwargs,
                )

                if verbose:
                    print(f"gvec: {gvec}")
                    print()
            else:
                if verbose:
                    print("Failed safety test ")
                    performance = np.nan

        else:
            n_constraints = len(spec.parse_trees)
            gvec = -np.inf*np.ones(n_constraints) # NSF is safe, so set g=-inf for all constraints
            if verbose:
                print("NSF")
            performance = np.nan

        # Clear out any cached data in the parse trees for the next trial.
        # This handles the case where solution_found=False.
        for parse_tree in spec_for_experiment.parse_trees:
            parse_tree.reset_base_node_dict(reset_data=True)
            
        # Write out file for this data_frac,trial_i combo
        data = [data_frac, trial_i, performance, passed_safety, gvec]
        colnames = ["data_frac", "trial_i", "performance", "passed_safety", "gvec"]
        self.write_trial_result(data, colnames, trial_dir, verbose=kwargs["verbose"])
        return

    def evaluate_constraint_functions(
        self, solution, constraint_eval_fns, constraint_eval_kwargs
    ):
        """Helper function for QSA() to evaluate
        the constraint functions to determine
        whether solution was safe on ground truth

        :param solution: The weights of the model found
                during candidate selection in a given trial
        :type solution: numpy ndarray

        :param constraint_eval_fns: List of functions
                to use to evaluate each constraint.
                An empty list results in using the parse
                tree to evaluate the constraints
        :type constraint_eval_fns: List(function)

        :param constraint_eval_kwargs: keyword arguments
                to pass to each constraint function
                in constraint_eval_fns
        :type constraint_eval_kwargs: dict
        """
        # Use safety test branch so the confidence bounds on
        # leaf nodes are not inflated
        gvals = []
        if constraint_eval_fns == []:
            """User did not provide their own functions
            to evaluate the constraints. Use the default:
            the parse tree has a built-in way to evaluate constraints.
            """
            constraint_eval_kwargs["theta"] = solution
            orig_dataset = constraint_eval_kwargs["spec_orig"].dataset
            spec_for_experiment = constraint_eval_kwargs["spec_for_experiment"]
            regime = constraint_eval_kwargs["regime"]
            if "eval_batch_size" in constraint_eval_kwargs:
                constraint_eval_kwargs["batch_size_safety"] = constraint_eval_kwargs[
                    "eval_batch_size"
                ]
            if regime == "supervised_learning":
                # X_test is the original images, after being passed through the trained headless model
                # for this trial. a.k.a the latent features
                X_test = constraint_eval_kwargs["X_test"]
                dataset_for_eval = SupervisedDataSet(
                    features=X_test,
                    labels=orig_dataset.labels,
                    sensitive_attrs=orig_dataset.sensitive_attrs,
                    num_datapoints=orig_dataset.num_datapoints,
                    meta=orig_dataset.meta)
                constraint_eval_kwargs["dataset"] = dataset_for_eval

            for parse_tree in spec_for_experiment.parse_trees:
                parse_tree.reset_base_node_dict(reset_data=True)
                parse_tree.evaluate_constraint(**constraint_eval_kwargs)

                g = parse_tree.root.value
                gvals.append(g)
                parse_tree.reset_base_node_dict(reset_data=True) # to clear out anything so the next trial has fresh data

        else:
            # User provided functions to evaluate constraints
            for eval_fn in constraint_eval_fns:
                g = eval_fn(solution)
                gvals.append(g)
        
        return np.array(gvals)

