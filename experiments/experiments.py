""" Module for running Seldonian Experiments """

import os
from operator import itemgetter
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import copy

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from seldonian.utils.io_utils import load_pickle
from seldonian.dataset import SupervisedDataSet, RLDataSet
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.spec import RLSpec
from seldonian.models.models import (
    LinearRegressionModel,
    BinaryLogisticRegressionModel,
    DummyClassifierModel,
    RandomClassifierModel,
)

from .experiment_utils import (
    batch_predictions,
    load_resampled_dataset,
    load_regenerated_episodes,
    prep_feat_labels,
    setup_SA_spec_for_exp,
    trial_arg_chunker
)

try:
    from fairlearn.reductions import ExponentiatedGradient
    from fairlearn.metrics import (
        MetricFrame,
        selection_rate,
        false_positive_rate,
        true_positive_rate,
        false_negative_rate,
    )
    from fairlearn.reductions import (
        DemographicParity,
        FalsePositiveRateParity,
        EqualizedOdds,
    )
except ImportError:
    print(
        "\nWARNING: The module 'fairlearn' was not imported. "
        "If you want to use the fairlearn baselines, then do:\n"
        "pip install fairlearn==0.7.0\n"
    )

import warnings
from seldonian.warnings.custom_warnings import *

warnings.filterwarnings("ignore", category=FutureWarning)

context = mp.get_context("spawn" if os.name == 'nt' else "fork")

class Experiment:
    def __init__(self, model_name, results_dir):
        """Base class for running experiments

        :param model_name: The string name of the baseline model,
                e.g 'logistic_regression'
        :type model_name: str

        :param results_dir: Parent directory for saving any
                experimental results
        :type results_dir: str
        """
        self.model_name = model_name
        self.results_dir = results_dir

    def aggregate_results(self, **kwargs):
        """Group together the data in each
        trial file into a single CSV file.
        """
        d = os.path.join(self.results_dir, f"{self.model_name}_results")
        os.makedirs(d, exist_ok=True)
        res_fname = os.path.join(d, f"{self.model_name}_results.csv")

        d_trial = os.path.join(
            self.results_dir, f"{self.model_name}_results", "trial_data"
        )
        df_list = []
        for data_frac in kwargs["data_fracs"]:
            for trial_i in range(kwargs["n_trials"]):
                filename = os.path.join(
                    d_trial, f"data_frac_{data_frac:.4f}_trial_{trial_i}.csv"
                )
                df = pd.read_csv(filename)
                df_list.append(df)

        res_df = pd.concat(df_list)
        res_df.to_csv(res_fname, index=False)
        if kwargs["verbose"]:
            print(f"Saved {res_fname}")
        return

    def write_trial_result(self, data, colnames, trial_dir, verbose=False):
        """Write out the results from a single trial
        to a file.

        :param data: The information to save
        :type data: List

        :param colnames: Names of the items in the list.
                These will comprise the header of the saved file
        :type colnames: List(str)

        :param trial_dir: The directory in which to save the file
        :type trial_dir: str

        :param verbose: if True, prints out saved filename
        :type verbose: bool
        """
        res_df = pd.DataFrame([data])
        res_df.columns = colnames
        data_frac, trial_i = data[0:2]

        fname = os.path.join(
            trial_dir, f"data_frac_{data_frac:.4f}_trial_{trial_i}.csv"
        )

        res_df.to_csv(fname, index=False)
        if verbose:
            print(f"Saved {fname}")
        return


class BaselineExperiment(Experiment):
    def __init__(self, baseline_model, results_dir):
        """Class for running baseline experiments
        against which to compare Seldonian Experiments

        :param model_name: The string name of the baseline model,
                e.g 'logistic_regression'
        :type model_name: str

        :param results_dir: Parent directory for saving any
                experimental results
        :type results_dir: str
        """
        self.baseline_model = baseline_model
        model_name = self.baseline_model.model_name
        super().__init__(model_name, results_dir)

    def run_experiment(self, **kwargs):
        """Run the baseline experiment"""
        partial_kwargs = {
            key: kwargs[key] for key in kwargs if key not in ["data_fracs", "n_trials"]
        }

        helper = partial(self.run_baseline_trial, **partial_kwargs)

        data_fracs = kwargs["data_fracs"]
        n_trials = kwargs["n_trials"]
        n_workers = kwargs["n_workers"]

        data_fracs_vec = np.array([x for x in data_fracs for y in range(n_trials)])
        trials_vec = np.array(
            [x for y in range(len(data_fracs)) for x in range(n_trials)]
        )

        if n_workers == 1:
            # run all trials synchronously
            for ii in range(len(data_fracs_vec)):
                data_frac = data_fracs_vec[ii]
                trial_i = trials_vec[ii]
                helper(data_frac, trial_i)

        elif n_workers > 1:
            # run trials asynchronously
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=context
            ) as ex:
                results = tqdm(
                    ex.map(helper, data_fracs_vec, trials_vec),
                    total=len(data_fracs_vec),
                )
                for exc in results:
                    if exc:
                        print(exc)
        else:
            raise ValueError(f"value of {n_workers} must be >=1 ")

        self.aggregate_results(**kwargs)

    def run_baseline_trial(self, data_frac, trial_i, **kwargs):
        """Run a trial of the baseline model. Currently only
        supports supervised learning experiments.

        :param data_frac: Fraction of overall dataset size to use
        :type data_frac: float

        :param trial_i: The index of the trial
        :type trial_i: int
        """

        spec = copy.deepcopy(kwargs["spec"])
        
        dataset = spec.dataset
        regime = dataset.regime
        parse_trees = spec.parse_trees
        (
            verbose,
            datagen_method,
            perf_eval_fn,
            perf_eval_kwargs,
            batch_epoch_dict,
            constraint_eval_fns,
            constraint_eval_kwargs,
        ) = itemgetter(
            "verbose",
            "datagen_method",
            "perf_eval_fn",
            "perf_eval_kwargs",
            "batch_epoch_dict",
            "constraint_eval_fns",
            "constraint_eval_kwargs",
        )(kwargs)

        if (
            batch_epoch_dict == {}
            and (spec.optimization_technique == "gradient_descent")
            and (spec.optimization_hyperparams["use_batches"] == True)
        ):
            warning_msg = (
                "WARNING: No batch_epoch_dict was provided. "
                "Each data_frac will use the same values "
                "for batch_size and n_epochs. "
                "This can have adverse effects",
                " " "especially for small values of data_frac.",
            )
            warnings.warn(warning_msg)

        d_trial = os.path.join(
            self.results_dir, f"{self.model_name}_results", "trial_data"
        )
        os.makedirs(d_trial, exist_ok=True)
        savename = os.path.join(
            d_trial, f"data_frac_{data_frac:.4f}_trial_{trial_i}.csv"
        )

        if os.path.exists(savename):
            if verbose:
                print(
                    f"Trial {trial_i} already run for "
                    f"this data_frac: {data_frac}. Skipping this trial. "
                )
            return

        ##############################################
        """ Setup for running baseline algorithm """
        ##############################################
        if regime == "supervised_learning":
            if datagen_method == "resample":
                trial_dataset, n_points = load_resampled_dataset(
                    self.results_dir, trial_i, data_frac, verbose=verbose
                )
            else:
                raise NotImplementedError(
                    f"datagen_method: {datagen_method} "
                    f"not supported for regime: {regime}"
                )

            features, labels = prep_feat_labels(trial_dataset, n_points)

            ####################################################
            """" Instantiate model and fit to resampled data """
            ####################################################
            X_test_baseline = perf_eval_kwargs["X"]
            baseline_model = copy.deepcopy(self.baseline_model)
            train_kwargs = {}
            pred_kwargs = {}
            try:
                if hasattr(baseline_model,"batch_epoch_dict"):
                    batch_size, n_epochs = baseline_model.batch_epoch_dict[data_frac]
                    train_kwargs["batch_size"] = batch_size
                    train_kwargs["n_epochs"] = n_epochs
                solution = baseline_model.train(features, labels, **train_kwargs)
                
                # predict the probabilities (e.g. 0.85) not the labels (e.g., 1) 
                if hasattr(baseline_model,"eval_batch_size"):
                    pred_kwargs["eval_batch_size"] = getattr(baseline_model,"eval_batch_size")
                    if hasattr(baseline_model,"N_output_classes"):
                        pred_kwargs["N_output_classes"] = getattr(baseline_model,"N_output_classes")

                    y_pred = batch_predictions(
                        model=baseline_model,
                        solution=solution,
                        X_test=X_test_baseline,
                        **pred_kwargs,
                    )
                else:
                    y_pred = baseline_model.predict(
                        solution, 
                        X_test_baseline, 
                        **pred_kwargs)
            except:
                if verbose: print("Error training baseline model. Returning NSF\n")
                solution = "NSF"
        elif regime == "reinforcement_learning":
            if datagen_method == "generate_episodes":
                trial_dataset = load_regenerated_episodes(
                    self.results_dir, trial_i, data_frac, spec.dataset.meta, verbose=verbose
                )
            else:
                raise NotImplementedError(
                    f"datagen_method: {datagen_method} "
                    f"not supported for regime: {regime}"
                )

            ####################################################
            """" Instantiate model and run it on resampled data """
            ####################################################
            baseline_model = copy.deepcopy(self.baseline_model)
            # train_kwargs = {}
            # pred_kwargs = {}
            try: 
                solution = baseline_model.train(trial_dataset)
                baseline_model.set_new_params(solution)
            except:
                solution = "NSF"

        #########################################################
        """" Calculate performance and safety on ground truth """
        #########################################################
        # Handle whether solution was found
        solution_found = True
        if type(solution) == str and solution == "NSF":
            solution_found = False

        if solution_found:
            if verbose: print("Solution was found. Calculating performance.\n")
            if regime == "supervised_learning":
                performance = perf_eval_fn(y_pred, **perf_eval_kwargs)
            elif regime == "reinforcement_learning":
                perf_eval_kwargs["model"] = baseline_model
                perf_eval_kwargs[
                    "hyperparameter_and_setting_dict"
                ] = kwargs["hyperparameter_and_setting_dict"]
                episodes_for_eval, performance = perf_eval_fn(**perf_eval_kwargs)

            if verbose:
                print(f"Performance = {performance}\n")

            # Determine whether this solution
            # violates any of the constraints
            # on the test dataset, which is the dataset from spec
            
            if regime == "supervised_learning":
                dataset_for_eval = dataset # the original one
            if regime == "reinforcement_learning":
                # Need to put the newly generated episodes into the new dataset
                constraint_eval_kwargs["episodes_for_eval"] = episodes_for_eval
                constraint_eval_kwargs["performance"] = performance
                dataset_for_eval = RLDataSet(
                    episodes=episodes_for_eval,
                    meta=dataset.meta
                )

            constraint_eval_kwargs["baseline_model"] = baseline_model
            constraint_eval_kwargs["dataset"] = dataset_for_eval
            constraint_eval_kwargs["parse_trees"] = parse_trees
            constraint_eval_kwargs["verbose"] = verbose

            gvec = self.evaluate_constraint_functions(
                solution=solution,
                constraint_eval_fns=constraint_eval_fns,
                constraint_eval_kwargs=constraint_eval_kwargs,
            )
        else:
            if verbose: print("NSF\n")
            # NSF is safe, so set g=-inf for all constraints
            n_constraints = len(spec.parse_trees)
            gvec = -np.inf * np.ones(
                n_constraints
            )  
            performance = np.nan

        # Write out file for this data_frac,trial_i combo
        # data = [data_frac, trial_i, performance, failed]
        # colnames = ["data_frac", "trial_i", "performance", "failed"]
        data = [data_frac, trial_i, performance, gvec]
        colnames = ["data_frac", "trial_i", "performance", "gvec"]
        self.write_trial_result(data, colnames, d_trial, verbose=kwargs["verbose"])
        return

    def evaluate_constraint_functions(
        self, solution, constraint_eval_fns, constraint_eval_kwargs
    ):
        """Helper function to evaluate
        the constraint functions to determine
        whether baseline solution was safe on ground truth

        :param solution: The weights of the model found
                during model training in a given trial
        :type solution: numpy ndarray
        :param constraint_eval_fns: List of functions
                to use to evaluate each constraint.
                An empty list (default) results in using the parse
                tree to evaluate the constraints
        :type constraint_eval_fns: List(function)
        :param constraint_eval_kwargs: keyword arguments
                to pass to each constraint function
                in constraint_eval_fns
        :type constraint_eval_kwargs: dict
        :return: a vector of g values for the constraints
        :rtype: np.ndarray
        """
        # Use safety test branch so the confidence bounds on
        # leaf nodes are not inflated
        gvals = []
        if constraint_eval_fns == []:
            parse_trees = constraint_eval_kwargs["parse_trees"]
            dataset_for_eval = constraint_eval_kwargs["dataset"]
            baseline_model = constraint_eval_kwargs["baseline_model"]
            if hasattr(baseline_model,"eval_batch_size"):
                batch_size_safety = getattr(baseline_model,"eval_batch_size")
            else:
                batch_size_safety = None
                
            for parse_tree in parse_trees:
                parse_tree.reset_base_node_dict(reset_data=True)
                parse_tree.evaluate_constraint(
                    theta=solution,
                    dataset=dataset_for_eval,
                    model=baseline_model,
                    regime=dataset_for_eval.regime,
                    branch="safety_test",
                    batch_size_safety=batch_size_safety,
                )
                g = parse_tree.root.value

                gvals.append(g)
                parse_tree.reset_base_node_dict(
                    reset_data=True
                )  # to clear out anything so the next trial has fresh data

        else:
            # User provided functions to evaluate constraints
            for eval_fn in constraint_eval_fns:
                g = eval_fn(solution, **constraint_eval_kwargs)
                gvals.append(g)
        return np.array(gvals)


class SeldonianExperiment(Experiment):
    def __init__(self, model_name, results_dir):
        """Class for running Seldonian experiments

        :param model_name: The string name of the Seldonian model,
                only option is currently: 'qsa' (quasi-Seldonian algorithm)
        :type model_name: str

        :param results_dir: Parent directory for saving any
                experimental results
        :type results_dir: str

        """
        super().__init__(model_name, results_dir)
        if self.model_name != "qsa":
            raise NotImplementedError(
                "Seldonian experiments for model: "
                f"{self.model_name} are not supported."
            )

    def run_experiment(self, **kwargs):
        """Run the Seldonian experiment"""

        n_workers = kwargs["n_workers"]
        trial_kwargs = {
            key: kwargs[key] for key in kwargs if key not in ["data_fracs", "n_trials"]
        }
        manager = mp.Manager()
        shared_namespace = manager.Namespace()

        # Store the trial_kwargs in the shared namespace
        shared_namespace.trial_kwargs = trial_kwargs

        data_fracs = kwargs["data_fracs"]
        n_trials = kwargs["n_trials"]

        if n_workers == 1:
            for data_frac in data_fracs:
                for trial_i in range(n_trials):
                    self.run_QSA_trial(data_frac, trial_i, **trial_kwargs)

        elif n_workers > 1:
            import itertools
            chunked_arg_list = trial_arg_chunker(data_fracs,n_trials,n_workers)
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=context
            ) as ex:
                results = tqdm(
                    ex.map(self.run_trials_par,chunked_arg_list,itertools.repeat(shared_namespace)),
                    total=len(chunked_arg_list),
                )
                for exc in results:
                    if exc:
                        print(exc)
        else:
            raise ValueError(f"n_workers value of {n_workers} must be >=1 ")

        self.aggregate_results(**kwargs)

    def run_trials_par(self, args_list, shared_namespace):
        for args in args_list:
            data_frac,trial_i = args
            self.run_QSA_trial(data_frac,trial_i,**shared_namespace.trial_kwargs)

    def run_QSA_trial(self, data_frac, trial_i, **kwargs):
        """Run a trial of the quasi-Seldonian algorithm

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
        if batch_epoch_dict == {} and spec.optimization_technique == "gradient_descent":
            warning_msg = (
                "WARNING: No batch_epoch_dict was provided. "
                "Each data_frac will use the same values "
                "for batch_size and n_epochs. "
                "This can have adverse effects, "
                "especially for small values of data_frac."
            )
            warnings.warn(warning_msg)
        regime = spec.dataset.regime

        trial_dir = os.path.join(self.results_dir, "qsa_results", "trial_data")

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

        spec_for_exp = setup_SA_spec_for_exp(
            spec=spec,
            regime=regime,
            results_dir=self.results_dir,
            trial_i=trial_i,
            data_frac=data_frac,
            datagen_method=datagen_method,
            batch_epoch_dict=batch_epoch_dict,
            kwargs=kwargs,
            perf_eval_kwargs=perf_eval_kwargs,
        )

        ################################
        """" Run Seldonian algorithm """
        ################################

        try:
            SA = SeldonianAlgorithm(spec_for_exp)
            passed_safety, solution = SA.run(write_cs_logfile=verbose, debug=verbose)
        except (ValueError, ZeroDivisionError):
            passed_safety = False
            solution = "NSF"

        if verbose:
            print(f"Solution from running seldonian algorithm: {solution}\n")


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
                    X_test = perf_eval_kwargs["X"]
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

                elif regime == "reinforcement_learning":
                    model = copy.deepcopy(SA.model)
                    model.policy.set_new_params(solution)
                    perf_eval_kwargs["model"] = model
                    perf_eval_kwargs[
                        "hyperparameter_and_setting_dict"
                    ] = kwargs["hyperparameter_and_setting_dict"]
                    episodes_new_policy, performance = perf_eval_fn(**perf_eval_kwargs)

                if verbose:
                    print(f"Performance = {performance}")
                    print(
                        "Determining whether solution "
                        "is actually safe on ground truth\n"
                    )
                ########################################
                """ Calculate safety on ground truth """
                ########################################

                if constraint_eval_fns == []:
                    constraint_eval_kwargs["model"] = model
                    constraint_eval_kwargs["spec_orig"] = spec
                    constraint_eval_kwargs["spec_for_exp"] = spec_for_exp
                    constraint_eval_kwargs["regime"] = regime
                    constraint_eval_kwargs["branch"] = "safety_test"
                    constraint_eval_kwargs["verbose"] = verbose

                if regime == "reinforcement_learning":
                    constraint_eval_kwargs["episodes_new_policy"] = episodes_new_policy
                    if "on_policy" not in constraint_eval_kwargs:
                        constraint_eval_kwargs["on_policy"] = True

                gvec = self.evaluate_constraint_functions(
                    solution=solution,
                    constraint_eval_fns=constraint_eval_fns,
                    constraint_eval_kwargs=constraint_eval_kwargs,
                )

                if verbose:
                    print(f"gvec: {gvec}\n")
            else:
                if verbose:
                    print("Failed safety test\n")
                    performance = np.nan

        else:  # solution_found=False
            n_constraints = len(spec.parse_trees)
            # NSF is safe, so set g=-inf for all constraints
            gvec = -np.inf * np.ones(n_constraints)
            if verbose:
                print("NSF\n")
            performance = np.nan

        # Clear out any cached data in the parse trees for the next trial.
        # This also handles the case where solution_found=False.
        for parse_tree in spec_for_exp.parse_trees:
            parse_tree.reset_base_node_dict(reset_data=True)

        # Write out file for this data_frac,trial_i combo
        data = [data_frac, trial_i, performance, passed_safety, gvec]
        colnames = ["data_frac", "trial_i", "performance", "passed_safety", "gvec"]
        self.write_trial_result(data, colnames, trial_dir, verbose=kwargs["verbose"])
        return

    def evaluate_constraint_functions(
        self, solution, constraint_eval_fns, constraint_eval_kwargs
    ):
        """Helper function for run_QSA_trial() to evaluate
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
        :return: a vector of g values for the constraints
        :rtype: np.ndarray
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
            spec_orig = constraint_eval_kwargs["spec_orig"]
            spec_for_exp = constraint_eval_kwargs["spec_for_exp"]
            regime = constraint_eval_kwargs["regime"]
            if "eval_batch_size" in constraint_eval_kwargs:
                constraint_eval_kwargs["batch_size_safety"] = constraint_eval_kwargs[
                    "eval_batch_size"
                ]
            if regime == "supervised_learning":
                # Use the original dataset as ground truth
                constraint_eval_kwargs["dataset"] = spec_orig.dataset

            elif regime == "reinforcement_learning":
                

                # Are we doing on policy or off policy evaluation?
                on_policy = constraint_eval_kwargs["on_policy"]
                if on_policy:
                    episodes_for_eval = constraint_eval_kwargs["episodes_new_policy"]
                else:
                    episodes_for_eval = spec_orig.dataset.episodes
                    
                dataset_for_eval = RLDataSet(
                    episodes=episodes_for_eval,
                    meta=spec_for_exp.dataset.meta,
                    regime=regime,
                )

                constraint_eval_kwargs["dataset"] = dataset_for_eval

            for parse_tree in spec_for_exp.parse_trees:
                parse_tree.reset_base_node_dict(reset_data=True)
                parse_tree.evaluate_constraint(**constraint_eval_kwargs)

                g = parse_tree.root.value
                gvals.append(g)
                parse_tree.reset_base_node_dict(
                    reset_data=True
                )  # to clear out anything so the next trial has fresh data

        else:
            # User provided functions to evaluate constraints
            assert len(constraint_eval_fns) == len(spec_for_exp.parse_trees)
            for eval_fn in constraint_eval_fns:
                g = eval_fn(solution, **constraint_eval_kwargs)
                gvals.append(g)

        return np.array(gvals)


class FairlearnExperiment(Experiment):
    def __init__(self, results_dir, fairlearn_epsilon_constraint):
        """Class for running Fairlearn experiments

        :param results_dir: Parent directory for saving any
                experimental results
        :type results_dir: str

        :param fairlearn_epsilon_constraint: The value of epsilon
                (the threshold) to use in the constraint
                to the Fairlearn model
        :type fairlearn_epsilon_constraint: float
        """
        super().__init__(
            results_dir=results_dir,
            model_name=f"fairlearn_eps{fairlearn_epsilon_constraint:.2f}",
        )

    def run_experiment(self, **kwargs):
        """Run the Fairlearn experiment"""
        n_workers = kwargs["n_workers"]
        partial_kwargs = {
            key: kwargs[key] for key in kwargs if key not in ["data_fracs", "n_trials"]
        }

        helper = partial(self.run_fairlearn_trial, **partial_kwargs)

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
                max_workers=n_workers, mp_context=context
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

    def run_fairlearn_trial(self, data_frac, trial_i, **kwargs):
        """Run a Fairlearn trial

        :param data_frac: Fraction of overall dataset size to use
        :type data_frac: float

        :param trial_i: The index of the trial
        :type trial_i: int
        """
        spec = kwargs["spec"]
        verbose = kwargs["verbose"]
        datagen_method = kwargs["datagen_method"]
        fairlearn_sensitive_feature_names = kwargs["fairlearn_sensitive_feature_names"]
        fairlearn_constraint_name = kwargs["fairlearn_constraint_name"]
        fairlearn_epsilon_constraint = kwargs["fairlearn_epsilon_constraint"]
        fairlearn_epsilon_eval = kwargs["fairlearn_epsilon_eval"]
        fairlearn_eval_kwargs = kwargs["fairlearn_eval_kwargs"]
        perf_eval_fn = kwargs["perf_eval_fn"]
        perf_eval_kwargs = kwargs["perf_eval_kwargs"]
        constraint_eval_fns = kwargs["constraint_eval_fns"]
        constraint_eval_kwargs = kwargs["constraint_eval_kwargs"]

        regime = spec.dataset.regime
        assert regime == "supervised_learning"

        trial_dir = os.path.join(
            self.results_dir,
            f"fairlearn_eps{fairlearn_epsilon_constraint:.2f}_results",
            "trial_data",
        )

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
        """ Setup for running Fairlearn algorithm """
        ##############################################

        if datagen_method == "resample":
            trial_dataset, n_points = load_resampled_dataset(
                self.results_dir, trial_i, data_frac
            )
        else:
            raise NotImplementedError(
                f"datagen_method: {datagen_method} "
                f"not supported for regime: {regime}"
            )

        # Prepare features and labels
        features, labels = prep_feat_labels(trial_dataset, n_points)

        sensitive_col_indices = [
            trial_dataset.sensitive_col_names.index(col)
            for col in fairlearn_sensitive_feature_names
        ]

        fairlearn_sensitive_features = np.squeeze(
            trial_dataset.sensitive_attrs[:, sensitive_col_indices]
        )[:n_points]

        ##############################################
        """" Run Fairlearn algorithm on trial data """
        ##############################################

        if fairlearn_constraint_name == "disparate_impact":
            fairlearn_constraint = DemographicParity(
                ratio_bound=fairlearn_epsilon_constraint
            )

        elif fairlearn_constraint_name == "demographic_parity":
            fairlearn_constraint = DemographicParity(
                difference_bound=fairlearn_epsilon_constraint
            )

        elif fairlearn_constraint_name == "predictive_equality":
            fairlearn_constraint = FalsePositiveRateParity(
                difference_bound=fairlearn_epsilon_constraint
            )

        elif fairlearn_constraint_name == "equalized_odds":
            fairlearn_constraint = EqualizedOdds(
                difference_bound=fairlearn_epsilon_constraint
            )

        elif fairlearn_constraint_name == "equal_opportunity":
            fairlearn_constraint = EqualizedOdds(
                difference_bound=fairlearn_epsilon_constraint
            )

        else:
            raise NotImplementedError(
                "Fairlearn constraints of type: "
                f"{fairlearn_constraint_name} "
                "is not supported."
            )

        classifier = LogisticRegression()

        mitigator = ExponentiatedGradient(classifier, fairlearn_constraint)
        solution_found = True

        try:
            mitigator.fit(
                features, labels, sensitive_features=fairlearn_sensitive_features
            )
            X_test_fairlearn = fairlearn_eval_kwargs[
                "X"
            ]  # same as X_test but drops the offset column
            y_pred = self.get_fairlearn_predictions(mitigator, X_test_fairlearn)
        except:
            print("Error when fitting. Returning NSF\n")
            solution_found = False
            performance = np.nan
        #########################################################
        """" Calculate performance and safety on ground truth """
        #########################################################

        if solution_found:
            fairlearn_eval_kwargs["model"] = mitigator
            # predict the class label, not the probability
            performance = perf_eval_fn(y_pred, **fairlearn_eval_kwargs)

            # Determine whether this solution
            # violates any of the constraints
            # on the test dataset
            fairlearn_eval_method = fairlearn_eval_kwargs["eval_method"]
            gvec = self.evaluate_constraint_function(
                y_pred=y_pred,
                test_labels=fairlearn_eval_kwargs["y"],
                fairlearn_constraint_name=fairlearn_constraint_name,
                epsilon_eval=fairlearn_epsilon_eval,
                eval_method=fairlearn_eval_method,
                sensitive_features=fairlearn_eval_kwargs["sensitive_features"],
            )
        else:  # solution_found=False
            n_constraints = len(spec.parse_trees)
            gvec = -np.inf * np.ones(
                n_constraints
            )  # NSF is safe, so set g=-inf for all constraints
            if verbose:
                print("NSF\n")

        # Write out file for this data_frac,trial_i combo
        data = [data_frac, trial_i, performance, gvec]
        colnames = ["data_frac", "trial_i", "performance", "gvec"]
        self.write_trial_result(data, colnames, trial_dir, verbose=kwargs["verbose"])
        return

    def get_fairlearn_predictions(self, mitigator, X_test_fairlearn):
        """
        Get the predicted labels from the fairlearn mitigator.
        The mitigator consists of potentially more than one predictor.
        For each predictor with non-zero weight, we figure out
        how many points to predict based on the weight of that predictor.
        Weights are normalized to 1 across all predictors.

        :param mitigator: The Fairlearn mitigator

        :param X_test_fairlearn: The test features from which
                to predict the labels
        """
        n_points_test = len(X_test_fairlearn)
        y_pred = np.zeros(n_points_test)
        assert len(mitigator.predictors_) == len(mitigator.weights_)
        start_index = 0
        for ii in range(len(mitigator.predictors_)):
            weight = mitigator.weights_[ii]
            if weight == 0:
                continue
            predictor = mitigator.predictors_[ii]
            n_points_this_predictor = int(round(weight * n_points_test))
            end_index = start_index + n_points_this_predictor
            X_test_this_predictor = X_test_fairlearn[start_index:end_index]

            probs = predictor.predict_proba(X_test_this_predictor)

            if probs.shape[1] == 1:
                # if only one class predicted it must be the positive class
                predictions = probs[:, 0]
            else:
                predictions = probs[:, 1]
            y_pred[start_index:end_index] = predictions
            start_index = end_index
        return y_pred

    def evaluate_constraint_function(
        self,
        y_pred,
        test_labels,
        fairlearn_constraint_name,
        epsilon_eval,
        eval_method="native",
        sensitive_features=[],
    ):
        """Evaluate the constraint function using the
        Fairlearn predictions

        :param y_pred: Predicted labels, same shape as test_labels
        :type y_pred: 1D array

        :param test_labels: True labels
        :type test_labels: 1D array

        :param fairlearn_constraint_name: The name of the constraint
        :type fairlearn_constraint_name: str

        :param epsilon_eval: The threshold in the constraint to use for
                evaluation
        :type epsilon_eval: float

        :param eval_method: The method for evaluating the constraint,
                two options: 'native' or 'two-groups'
        :type eval_method: str, defaults to 'native'

        :param sensitive_features: List of column names that are considered
                sensitive in the Fairlearn dataset
        :type sensitive_features: List(str)
        """
        failed = False
        if fairlearn_constraint_name == "demographic_parity":
            # g = abs((PR | ATR1) - (PR | ATR2)) - eps
            PR_frame = MetricFrame(
                metrics=selection_rate,
                y_true=test_labels,
                y_pred=y_pred >= 0.5,
                sensitive_features=sensitive_features,
            )
            PR_grouped = PR_frame.by_group
            PR_group1 = PR_grouped.iloc[0]
            if eval_method == "native":
                PR_overall = PR_frame.overall
                g = abs(PR_group1 - PR_overall) - epsilon_eval
            elif eval_method == "two-groups":
                PR_group2 = PR_grouped.iloc[1]
                g = abs(PR_group1 - PR_group2) - epsilon_eval

        elif fairlearn_constraint_name == "predictive_equality":
            # g = abs((FPR | ATR1) - (FPR | ATR2)) - eps
            FPR_frame = MetricFrame(
                metrics=false_positive_rate,
                y_true=test_labels,
                y_pred=y_pred >= 0.5,
                sensitive_features=sensitive_features,
            )
            FPR_grouped = FPR_frame.by_group
            FPR_group1 = FPR_grouped.iloc[0]
            if eval_method == "native":
                FPR_overall = FPR_frame.overall
                g = abs(FPR_group1 - FPR_overall) - epsilon_eval
            elif eval_method == "two-groups":
                FPR_group2 = FPR_grouped.iloc[1]
                g = abs(FPR_group1 - FPR_group2) - epsilon_eval

        elif fairlearn_constraint_name == "disparate_impact":
            # g = epsilon - min((PR | ATR1)/(PR | ATR2),(PR | ATR2)/(PR | ATR1))
            PR_frame = MetricFrame(
                metrics=selection_rate,
                y_true=test_labels,
                y_pred=y_pred >= 0.5,
                sensitive_features=sensitive_features,
            )

            PR_grouped = PR_frame.by_group
            PR_group1 = PR_grouped.iloc[0]
            if eval_method == "native":
                PR_overall = PR_frame.overall
                g = epsilon_eval - min(PR_group1 / PR_overall, PR_overall / PR_group1)
            elif eval_method == "two-groups":
                PR_group2 = PR_grouped.iloc[1]
                g = epsilon_eval - min(PR_group1 / PR_group2, PR_group2 / PR_group1)

        elif fairlearn_constraint_name == "equalized_odds":
            # g = abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) - epsilon
            FPR_frame = MetricFrame(
                metrics=false_positive_rate,
                y_true=test_labels,
                y_pred=y_pred >= 0.5,
                sensitive_features=sensitive_features,
            )
            FPR_grouped = FPR_frame.by_group
            FPR_group1 = FPR_grouped.iloc[0]

            FNR_frame = MetricFrame(
                metrics=false_negative_rate,
                y_true=test_labels,
                y_pred=y_pred >= 0.5,
                sensitive_features=sensitive_features,
            )
            FNR_grouped = FNR_frame.by_group
            FNR_group1 = FNR_grouped.iloc[0]

            if eval_method == "native":
                FPR_overall = FPR_frame.overall
                FNR_overall = FNR_frame.overall
                g = (
                    abs(FPR_group1 - FPR_overall)
                    + abs(FNR_group1 - FNR_overall)
                    - epsilon_eval
                )
            elif eval_method == "two-groups":
                FPR_group2 = FPR_grouped.iloc[1]
                FNR_group2 = FNR_grouped.iloc[1]
                g = (
                    abs(FPR_group1 - FPR_group2)
                    + abs(FNR_group1 - FNR_group2)
                    - epsilon_eval
                )

        elif fairlearn_constraint_name == "equal_opportunity":
            # g = abs((FNR | [M]) - (FNR | [F])) - epsilon

            FNR_frame = MetricFrame(
                metrics=false_negative_rate,
                y_true=test_labels,
                y_pred=y_pred >= 0.5,
                sensitive_features=sensitive_features,
            )
            FNR_grouped = FNR_frame.by_group
            FNR_group1 = FNR_grouped.iloc[0]

            if eval_method == "native":
                FNR_overall = FNR_frame.overall
                g = abs(FNR_group1 - FNR_overall) - epsilon_eval
            elif eval_method == "two-groups":
                FNR_group2 = FNR_grouped.iloc[1]
                g = abs(FNR_group1 - FNR_group2) - epsilon_eval
        else:
            raise NotImplementedError(
                "Evaluation for Fairlearn constraints of type: "
                f"{fairlearn_constraint.short_name} "
                "is not supported."
            )
        return np.array([g])
