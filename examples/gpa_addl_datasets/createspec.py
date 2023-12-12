import os
import autograd.numpy as np  # Thinly-wrapped version of Numpy
import pytest

from seldonian.parse_tree.parse_tree import *
from seldonian.utils.io_utils import load_json, load_pickle
from seldonian.utils.tutorial_utils import generate_data
from seldonian.dataset import *
from seldonian.spec import *
from seldonian.models import objectives
from seldonian.models.models import *

if __name__ == "__main__":
    base_path = "../../../engine-repo-dev"
    data_pth = os.path.join(base_path,"static/datasets/supervised/GPA/gpa_classification_dataset.csv")
    metadata_pth = os.path.join(base_path,"static/datasets/supervised/GPA/metadata_classification.json")

    metadata_dict = load_json(metadata_pth)
    all_col_names = metadata_dict["all_col_names"]
    regime = metadata_dict["regime"]
    sub_regime = metadata_dict["sub_regime"]
    sensitive_col_names = metadata_dict["sensitive_col_names"]

    verbose=True
    regime = "supervised_learning"

    model = BinaryLogisticRegressionModel()

    # Mean squared error
    primary_objective = objectives.binary_logistic_loss

    # Load dataset from file
    loader = DataSetLoader(regime=regime)

    # The primary dataset is just the original dataset
    primary_dataset = loader.load_supervised_dataset(
        filename=data_pth, metadata_filename=metadata_pth, file_type="csv"
    )
    primary_meta = primary_dataset.meta

    # Now make a dataset to use for bounding the base nodes
    # Take 80% of the original data
    orig_features = primary_dataset.features
    orig_labels = primary_dataset.labels
    orig_sensitive_attrs = primary_dataset.sensitive_attrs
    num_datapoints_new = int(round(len(orig_features)*0.8))
    rand_indices = np.random.choice(
        a=range(len(orig_features)),
        size=num_datapoints_new,
        replace=False
    )
    new_features = orig_features[rand_indices]
    new_labels = orig_labels[rand_indices]
    new_sensitive_attrs = orig_sensitive_attrs[rand_indices]
    new_meta = SupervisedMetaData(
        sub_regime=sub_regime,
        all_col_names=all_col_names,
        feature_col_names=primary_meta.feature_col_names,
        label_col_names=primary_meta.label_col_names,
        sensitive_col_names=sensitive_col_names,
    )
    new_dataset = SupervisedDataSet(
        features=new_features,
        labels=new_labels,
        sensitive_attrs=new_sensitive_attrs,
        num_datapoints=num_datapoints_new,
        meta=new_meta
    )


    constraint_strs = ['0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'] 
    deltas = [0.05]

    # For each constraint, make a parse tree 
    parse_trees = []
    for ii in range(len(constraint_strs)):
        constraint_str = constraint_strs[ii]
        delta = deltas[ii]
        # Create parse tree object
        parse_tree = ParseTree(
            delta=delta,
            regime="supervised_learning",
            sub_regime=sub_regime,
            columns=sensitive_col_names,
        )

        parse_tree.build_tree(constraint_str=constraint_str)
        parse_trees.append(parse_tree)

    # For each base node in each parse_tree, 
    # add this new dataset to additional_datasets dictionary
    # It is possible that when a parse tree is built, 
    # the constraint string it stores is different than the one that 
    # was used as input. This is because the parser may simplify the expression
    # Therefore, we want to use the constraint string attribute of the built parse 
    # tree as the key to the additional_datasets dict.


    additional_datasets = {}
    for pt in parse_trees:
        additional_datasets[pt.constraint_str] = {}
        base_nodes_this_tree = list(pt.base_node_dict.keys())
        for bn in base_nodes_this_tree:
            additional_datasets[pt.constraint_str][bn] = {
                "dataset": new_dataset
            }


    frac_data_in_safety = 0.6

    def initial_solution_fn(m, x, y):
        return m.fit(x, y)

    # Create spec object
    spec = SupervisedSpec(
        dataset=primary_dataset,
        additional_datasets=additional_datasets,
        model=model,
        parse_trees=parse_trees,
        sub_regime=sub_regime,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=primary_objective,
        use_builtin_primary_gradient_fn=True,
        initial_solution_fn=initial_solution_fn,
        optimization_technique="gradient_descent",
        optimizer="adam",
        optimization_hyperparams={
            "lambda_init": np.array([0.5]),
            "alpha_theta": 0.01,
            "alpha_lamb": 0.01,
            "beta_velocity": 0.9,
            "beta_rmsprop": 0.95,
            "num_iters": 1000,
            "use_batches": False,
            "gradient_library": "autograd",
            "hyper_search": None,
            "verbose": verbose,
        },
    )

    save_dir = './specfiles'
    os.makedirs(save_dir,exist_ok=True)
    spec_save_name = os.path.join(save_dir, "spec_disparate_impact_0.8.pkl")
    save_pickle(spec_save_name, spec, verbose=verbose)



