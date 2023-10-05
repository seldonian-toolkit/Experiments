import autograd.numpy as np   # Thinly-wrapped version of Numpy


from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from examples.headless_mnist.headonly_model import CNNHead
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)
from seldonian.utils.io_utils import save_pickle

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    torch.manual_seed(0)
    accuracy_threshold = 0.95 # Constraint is ACC >= accuracy_threshold
    regime='supervised_learning'
    sub_regime='multiclass_classification'
    data_folder = '../../../notebooks/data' # change this to where you want to download MNIST data
    train_data = datasets.MNIST(
        root = data_folder,
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = data_folder,
        train = False,                         
        transform = ToTensor(), 
        download = True,            
    )
    # Combine train and test data into a single tensor of 70,000 examples
    all_data = torch.vstack((train_data.data,test_data.data))
    all_targets = torch.hstack((train_data.targets,test_data.targets))
    N=len(all_targets) 
    assert N == 70000
    frac_data_in_safety = 0.5
    features = np.array(all_data.reshape(N,1,28,28),dtype='float32')/255.0
    labels = np.array(all_targets) # these are 1D so don't need to reshape them

    meta_information = {}
    meta_information['feature_col_names'] = ['img']
    meta_information['label_col_names'] = ['label']
    meta_information['sensitive_col_names'] = []
    meta_information['sub_regime'] = sub_regime

    dataset = SupervisedDataSet(
        features=features,
        labels=labels,
        sensitive_attrs=[],
        num_datapoints=N,
        meta_information=meta_information)

    constraint_strs = [f'ACC >= {accuracy_threshold}']
    deltas = [0.05] 

    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime)
    device = torch.device("cpu")
    model = CNNHead(device=device) # head gets run on CPU

    initial_solution_fn = model.get_model_params
    
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        frac_data_in_safety=frac_data_in_safety,
        primary_objective=objectives.multiclass_logistic_loss,
        use_builtin_primary_gradient_fn=False,
        sub_regime=sub_regime,
        initial_solution_fn=initial_solution_fn,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 0.001,
            'alpha_lamb'    : 0.01,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : True,
            'batch_size'    : 150,
            'n_epochs'      : 5,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : False,
        },
    )

    save_pickle(
        f"data/spec/headless_mnist_accuracy_{accuracy_threshold}.pkl",
        spec,
        verbose=True)
    