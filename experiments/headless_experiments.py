""" Module for running Seldonian Experiments """

from .experiments import Experiment 

from .utils import headless_utils

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

        parse_trees = spec.parse_trees
        dataset = spec.dataset

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

        # Obtain latent features by training the full model 
        # and then passing the data through a headless version of this model

        latent_features = headless_utils.generate_latent_features(
            full_model=full_model,
            headless_model=headless_model,
            head_layer_names=head_layer_names,
            orig_features=features,
            labels=labels, 
            frac_data_in_safety=spec.frac_data_in_safety, 
            candidate_batch_size=candidate_batch_size, 
            safety_batch_size=safety_batch_size,
            loss_func=nn.CrossEntropyLoss(),
            learning_rate=0.001,
            num_epochs=5,
            device=torch.device("cpu"))

        dataset_for_experiment = SupervisedDataSet(
            features=latent_features,
            labels=labels,
            sensitive_attrs=sensitive_attrs,
            num_datapoints=n_points,
            meta_information=resampled_dataset.meta_information,
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

        failed = False  # flag for whether we were actually safe on test set

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

                if regime == "reinforcement_learning":
                    model = copy.deepcopy(SA.model)
                    model.policy.set_new_params(solution)
                    perf_eval_kwargs["model"] = model
                    perf_eval_kwargs[
                        "hyperparameter_and_setting_dict"
                    ] = hyperparameter_and_setting_dict
                    episodes_for_eval, performance = perf_eval_fn(**perf_eval_kwargs)
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
                    constraint_eval_kwargs["spec_orig"] = spec
                    constraint_eval_kwargs["spec_for_experiment"] = spec_for_experiment
                    constraint_eval_kwargs["regime"] = regime
                    constraint_eval_kwargs["branch"] = "safety_test"
                    constraint_eval_kwargs["verbose"] = verbose

                if regime == "reinforcement_learning":
                    constraint_eval_kwargs["episodes_for_eval"] = episodes_for_eval

                failed = self.evaluate_constraint_functions(
                    solution=solution,
                    constraint_eval_fns=constraint_eval_fns,
                    constraint_eval_kwargs=constraint_eval_kwargs,
                )

                if verbose:
                    if failed:
                        print("Solution was not actually safe on ground truth!")
                    else:
                        print("Solution was safe on ground truth")
                    print()
            else:
                if verbose:
                    print("Failed safety test ")
                    performance = np.nan

        else:
            if verbose:
                print("NSF")
            performance = np.nan
        # Write out file for this data_frac,trial_i combo
        data = [data_frac, trial_i, performance, passed_safety, failed]
        colnames = ["data_frac", "trial_i", "performance", "passed_safety", "failed"]
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
        failed = False
        if constraint_eval_fns == []:
            """User did not provide their own functions
            to evaluate the constraints. Use the default:
            the parse tree has a built-in way to evaluate constraints.
            """
            constraint_eval_kwargs["theta"] = solution
            spec_orig = constraint_eval_kwargs["spec_orig"]
            spec_for_experiment = constraint_eval_kwargs["spec_for_experiment"]
            regime = constraint_eval_kwargs["regime"]
            if "eval_batch_size" in constraint_eval_kwargs:
                constraint_eval_kwargs["batch_size_safety"] = constraint_eval_kwargs[
                    "eval_batch_size"
                ]
            if regime == "supervised_learning":
                # Use the original dataset as ground truth
                constraint_eval_kwargs["dataset"] = spec_orig.dataset

            elif regime == "reinforcement_learning":
                episodes_for_eval = constraint_eval_kwargs["episodes_for_eval"]

                dataset_for_eval = RLDataSet(
                    episodes=episodes_for_eval,
                    meta_information=spec_for_experiment.dataset.meta_information,
                    regime=regime,
                )

                constraint_eval_kwargs["dataset"] = dataset_for_eval

            for parse_tree in spec_for_experiment.parse_trees:
                parse_tree.reset_base_node_dict(reset_data=True)
                parse_tree.evaluate_constraint(**constraint_eval_kwargs)

                g = parse_tree.root.value
                if g > 0 or np.isnan(g):
                    failed = True

        else:
            # User provided functions to evaluate constraints
            for eval_fn in constraint_eval_fns:
                g = eval_fn(solution)
                if g > 0 or np.isnan(g):
                    failed = True
        return failed


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

        parse_trees = spec.parse_trees
        dataset = spec.dataset

        ##############################################
        """ Setup for running Fairlearn algorithm """
        ##############################################

        if datagen_method == "resample":
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
        else:
            raise NotImplementedError(
                f"datagen_method: {datagen_method} "
                f"not supported for regime: {regime}"
            )

        # Prepare features and labels
        features = resampled_dataset.features
        labels = resampled_dataset.labels
        # Only use first n_points for this trial
        if type(features) == list:
            features = [x[:n_points] for x in features]
        else:
            features = features[:n_points]
        labels = labels[:n_points]

        sensitive_col_indices = [
            resampled_dataset.sensitive_col_names.index(col)
            for col in fairlearn_sensitive_feature_names
        ]

        fairlearn_sensitive_features = np.squeeze(
            resampled_dataset.sensitive_attrs[:, sensitive_col_indices]
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
            print("Error when fitting. Returning NSF")
            solution_found = False
            performance = np.nan
        #########################################################
        """" Calculate performance and safety on ground truth """
        #########################################################
        if solution_found:
            fairlearn_eval_kwargs["model"] = mitigator

            performance = perf_eval_fn(y_pred, **fairlearn_eval_kwargs)

        if verbose:
            print(f"Performance = {performance}")

        ########################################
        """ Calculate safety on ground truth """
        ########################################
        if verbose:
            print("Determining whether solution " "is actually safe on ground truth")

        # predict the class label, not the probability
        # Determine whether this solution
        # violates any of the constraints
        # on the test dataset
        failed = False
        if solution_found:
            fairlearn_eval_method = fairlearn_eval_kwargs["eval_method"]
            failed = self.evaluate_constraint_function(
                y_pred=y_pred,
                test_labels=fairlearn_eval_kwargs["y"],
                fairlearn_constraint_name=fairlearn_constraint_name,
                epsilon_eval=fairlearn_epsilon_eval,
                eval_method=fairlearn_eval_method,
                sensitive_features=fairlearn_eval_kwargs["sensitive_features"],
            )
        else:
            failed = False

        if failed:
            print("Fairlearn trial UNSAFE on ground truth")
        else:
            print("Fairlearn trial SAFE on ground truth")
        # Write out file for this data_frac,trial_i combo
        data = [data_frac, trial_i, performance, failed]

        colnames = ["data_frac", "trial_i", "performance", "failed"]
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
        print(f"Evaluating constraint for: {fairlearn_constraint_name}")
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

        print(f"g (fairlearn) = {g}")
        if g > 0 or np.isnan(g):
            failed = True

        return failed
