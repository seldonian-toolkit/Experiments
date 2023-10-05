import autograd.numpy as np
import os
import pandas as pd
from seldonian.parse_tree.parse_tree import ParseTree, make_parse_trees_from_constraints

from seldonian.utils.io_utils import load_json, save_pickle, load_pickle
from seldonian.spec import SupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel,
)
from seldonian.models import objectives
from seldonian.dataset import SupervisedDataSet,SupervisedMetaData


save_dir = "data/spec"
os.makedirs(save_dir, exist_ok=True)

# Define task
regime = "supervised_learning"
sub_regime = "classification"

# Load features and labels
DATA_PATH = "data/proc_data/data.pkl"
LIWC = [
    "words_per_sentence",
    "six_plus_words",
    "word_count",
    "function",
    "pronoun",
    "ppron",
    "i",
    "we",
    "you",
    "she",
    "he",
    "they",
    "ipron",
    "article",
    "prep",
    "auxverb",
    "adverb",
    "conj",
    "negate",
    "verb",
    "adj",
    "compare",
    "interrog",
    "number",
    "quant",
    "affect",
    "posemo",
    "negemo",
    "anx",
    "anger",
    "sad",
    "social",
    "family",
    "friend",
    "female",
    "male",
    "cogproc",
    "insight",
    "cause",
    "discrep",
    "tentat",
    "certain",
    "differ",
    "percept",
    "see",
    "hear",
    "feel",
    "bio",
    "body",
    "health",
    "sexual",
    "ingest",
    "drives",
    "affiliation",
    "achieve",
    "power",
    "reward",
    "risk",
    "focuspast",
    "focuspresent",
    "focusfuture",
    "relativ",
    "motion",
    "space",
    "time",
    "work",
    "leisure",
    "home",
    "money",
    "relig",
    "death",
    "informal",
    "swear",
    "netspeak",
    "assent",
    "nonflu",
    "filler",
]

JOB = ["job_" + str(i) for i in range(20)]
STATE = ["state_" + str(i) for i in range(20)]
PARTY = ["party_" + str(i) for i in range(20)]
CONTEXT = ["context_" + str(i) for i in range(20)]

feature_attr = LIWC + JOB + STATE + PARTY + CONTEXT
sensitive_attr = ["democrat", "republican"]
label_attr = ["label"]

data = pd.read_pickle(DATA_PATH)
features = data.loc[:, feature_attr].values.astype(float)
sensitive = data.loc[:, sensitive_attr].values.astype(int)
label = data.loc[:, label_attr].values.squeeze().astype(int)

# Construct meta data
meta_information = {}
meta = SupervisedMetaData(
        sub_regime, 
        all_col_names=feature_attr+sensitive_attr+label_attr, 
        feature_col_names=feature_attr,
        label_col_names=label_attr,
        sensitive_col_names=sensitive_attr)
# meta_information["feature_col_names"] = feature_attr
# meta_information["label_col_names"] = label_attr
# meta_information["sensitive_col_names"] = sensitive_attr
# meta_information["sub_regime"] = sub_regime

# Create dataset object
dataset = SupervisedDataSet(
    features=features,
    labels=label,
    sensitive_attrs=sensitive,
    num_datapoints=features.shape[0],
    meta=meta,
)

# Set the primary objective to be log loss
primary_objective = objectives.binary_logistic_loss

# Use logistic regression model
model = LogisticRegressionModel()

# Behavioral constraints
constraint_names = [
    "disparate_impact",
    "predictive_equality",
    "equal_opportunity",
    "overall_accuracy_equality",
]
epsilons = [0.2, 0.1, 0.05]
deltas = [0.05]

for constraint_name in constraint_names:
    for epsilon in epsilons:
        # Define behavioral constraints
        if constraint_name == "disparate_impact":
            constraint_strs = [
                f"min((PR | [democrat])/(PR | [republican]),(PR | [republican])/(PR | [democrat])) >= {1-epsilon}"
            ]
        elif constraint_name == "equal_opportunity":
            constraint_strs = [
                f"min((FNR | [democrat])/(FNR | [republican]),(FNR | [republican])/(FNR | [democrat])) >= {1-epsilon}"
            ]
        elif constraint_name == "predictive_equality":
            constraint_strs = [
                f"min((FPR | [democrat])/(FPR | [republican]),(FPR | [republican])/(FPR | [democrat])) >= {1-epsilon}"
            ]
        elif constraint_name == "overall_accuracy_equality":
            constraint_strs = [
                f"min((ACC | [democrat])/(ACC | [republican]),(ACC | [republican])/(ACC | [democrat])) >= {1-epsilon}"
            ]

        # For each constraint, make a parse tree
        parse_trees = make_parse_trees_from_constraints(
            constraint_strs,
            deltas,
            regime=regime,
            sub_regime=sub_regime,
            columns=sensitive_attr,
        )

        # Save spec object, using defaults where necessary
        spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            parse_trees=parse_trees,
            sub_regime=sub_regime,
            frac_data_in_safety=0.3,
            primary_objective=primary_objective,
            initial_solution_fn=model.fit,
            use_builtin_primary_gradient_fn=True,
            optimization_technique="gradient_descent",
            optimizer="adam",
            optimization_hyperparams={
                "lambda_init": np.array([0.5]),
                "alpha_theta": 0.0005,
                "alpha_lamb": 0.0005,
                "beta_velocity": 0.9,
                "beta_rmsprop": 0.95,
                "use_batches": False,
                "num_iters": 6000,
                "gradient_library": "autograd",
                "hyper_search": None,
                "verbose": True,
            },
        )

        spec_save_name = os.path.join(
            save_dir, f"lie_detection_{constraint_name}_{epsilon}.pkl"
        )
        save_pickle(spec_save_name, spec)
        print(f"Saved Spec object to: {spec_save_name}")
