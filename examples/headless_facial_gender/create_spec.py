import autograd.numpy as np   # Thinly-wrapped version of Numpy

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from examples.headless_facial_gender.headonly_model import CNNHead
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)
from seldonian.utils.io_utils import load_pickle,save_pickle

import torch

if __name__ == "__main__":
    torch.manual_seed(0)
    os.makedirs('./data/spec',exist_ok=True)
    regime='supervised_learning'
    sub_regime='classification'
    N=23700
    savename_features = './data/proc/features.pkl'
    savename_labels = './data/proc/labels.pkl'
    savename_sensitive_attrs = './data/proc/sensitive_attrs.pkl'
    features = load_pickle(savename_features)
    labels = load_pickle(savename_labels)
    sensitive_attrs = load_pickle(savename_sensitive_attrs)
    
    assert len(features) == N
    assert len(labels) == N
    assert len(sensitive_attrs) == N
    frac_data_in_safety = 0.5
    sensitive_col_names = ['M','F']

    meta_information = {}
    meta_information['feature_col_names'] = ['img']
    meta_information['label_col_names'] = ['label']
    meta_information['sensitive_col_names'] = sensitive_col_names
    meta_information['sub_regime'] = sub_regime
    
    dataset = SupervisedDataSet(
        features=features,
        labels=labels,
        sensitive_attrs=sensitive_attrs,
        num_datapoints=N,
        meta_information=meta_information)

    epsilon = 0.8
    constraint_strs = [f'min((ACC | [M])/(ACC | [F]),(ACC | [F])/(ACC | [M])) >= {epsilon}']
    deltas = [0.05] 
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime,columns=sensitive_col_names)

    device = torch.device("cpu")
    model = CNNHead(device=device) # head gets run on CPU

    initial_solution_fn = model.get_model_params
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=objectives.binary_logistic_loss,
        use_builtin_primary_gradient_fn=False,
        sub_regime=sub_regime,
        initial_solution_fn=initial_solution_fn,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.01,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : False,
            'num_iters'     : 1200,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : False,
        },
        batch_size_safety=2000
    )

    save_pickle(
        f"data/spec/headless_facial_gender_overall_accuracy_equality_{epsilon}.pkl",
        spec,
        verbose=True)
    