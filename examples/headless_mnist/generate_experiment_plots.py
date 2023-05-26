## Run headless mnist experiments (not yet documented on the website)

### Epsilon is the accuracy threshold:
# 

import argparse
import numpy as np
import math
import os
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.headless_example import HeadlessExample
from experiments.experiment_utils import (make_batch_epoch_dict_min_sample_repeat)
from experiments.perf_eval_funcs import (multiclass_accuracy,
    multiclass_logistic_loss)
from experiments.headless_utils import make_data_loaders
from seldonian.utils.io_utils import load_pickle
from examples.headless_mnist.full_model import CNN
from examples.headless_mnist.headless_model import CNN_headless

import torch
import torch.nn as nn

def headless_mnist_example(
    spec_rootdir,
    results_base_dir,
    accuracy_thresholds=[0.95],
    n_trials=50,
    data_fracs=np.logspace(-3,0,15),
    baselines = [],
    performance_metric="accuracy",
    n_workers=1,
    pretraining_device=torch.device('mps'),
    verbose=False,
):  
    full_pretraining_model = CNN()
    headless_pretraining_model = CNN_headless()
    head_layer_names = ['out.weight','out.bias']
    
    data_fracs = np.array(data_fracs)
    # Ensure that number of iterations in pretraining
    # is max(niter_min_pretraining,# of iterations 
    # s.t. each sample is seen num_repeats times)
    niter_min_pretraining=25 # min iterations we want in each run. 
    num_repeats=5
    batch_size_pretraining=150
    N_candidate_max=35000
    batch_epoch_dict_pretraining = make_batch_epoch_dict_min_sample_repeat(
        niter_min_pretraining,
        data_fracs,
        N_candidate_max,
        batch_size_pretraining,
        num_repeats)
    # Ensure that the number of iterations in candidate selection
    # does not change with data_frac
    niter_min=1000 # how many iterations we want in each run. 
    num_repeats=5
    batch_size=150
    batch_epoch_dict = make_batch_epoch_dict_min_sample_repeat(
        niter_min,
        data_fracs,
        N_candidate_max,
        batch_size,
        num_repeats)
    # The function that will calculate performance
    if performance_metric == "accuracy":
        perf_eval_fn = multiclass_accuracy
        performance_yscale='linear'
    elif performance_metric == 'log_loss':
        perf_eval_fn = multiclass_logistic_loss
        performance_yscale='log'
    else:
        raise NotImplementedError(
            f"Performance metric {performance_metric} not supported for this example")

    for accuracy_threshold in accuracy_thresholds:
        specfile = os.path.join(
            spec_rootdir,
            f"headless_mnist_accuracy_{accuracy_threshold}.pkl"
        )
        try: 
            spec = load_pickle(specfile)
        except ModuleNotFoundError:
            os.chdir()
        results_dir = os.path.join(results_base_dir,
            f"headless_mnist_accuracy_{accuracy_threshold}_perf_{performance_metric}")
        plot_savename = os.path.join(
            results_dir, f"accuracy_{accuracy_threshold}_perf_{performance_metric}.pdf"
        )
        # Set up kwargs that will be passed into the perf_eval_fn function
        dataset = spec.dataset
        test_features = dataset.features
        test_labels = dataset.labels

        # Make data loaders out of features and labels
        # Batch sizes are only used for single forward pass,
        # so for memory concerns only, not for training purposes
        test_data_loaders = make_data_loaders(
            test_features, 
            test_labels, 
            spec.frac_data_in_safety, 
            candidate_batch_size=1000, 
            safety_batch_size=1000,
        )
        perf_eval_kwargs = {
            "test_data_loaders": test_data_loaders,
            "y": test_labels,
            "eval_batch_size": 2000, # how many latent features are passed through perf_eval_fn at a time. Lower if running into memory issues
            "N_output_classes": 10
        }
        ex = HeadlessExample(spec=spec)

        ex.run(
            full_pretraining_model=full_pretraining_model,
            headless_pretraining_model=headless_pretraining_model,
            head_layer_names=head_layer_names,
            latent_feature_shape=(1568,),
            loss_func_pretraining=nn.CrossEntropyLoss(),
            learning_rate_pretraining=0.001,
            pretraining_device=pretraining_device,
            batch_epoch_dict_pretraining=batch_epoch_dict_pretraining,
            safety_batch_size_pretraining=1000,
            n_trials=n_trials,
            data_fracs=data_fracs,
            results_dir=results_dir,
            perf_eval_fn=perf_eval_fn,
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs={},
            n_workers=n_workers,
            batch_epoch_dict=batch_epoch_dict,
            datagen_method="resample",
            verbose=verbose,
            baselines=baselines,
            performance_label=performance_metric,
            performance_yscale=performance_yscale,
            plot_savename=plot_savename,
            plot_fontsize=12,
            legend_fontsize=8
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--accuracy_threshold', help='Accuracy >= accuracy_threshold', required=True)
    parser.add_argument('--n_trials', help='Number of trials to run', required=True)
    parser.add_argument('--n_workers', help='Number of workers to use', required=True)
    parser.add_argument('--verbose', help='verbose', action="store_true")

    args = parser.parse_args()

    accuracy_threshold = float(args.accuracy_threshold)
    n_trials = int(args.n_trials)
    n_workers = int(args.n_workers)
    verbose = args.verbose

    # if include_baselines:
    #     baselines = ["random_classifier","logistic_regression"]
    # else:
    #     baselines = []
    performance_metric="accuracy"

    results_base_dir = f"./results"

    headless_mnist_example(
        spec_rootdir="data/spec/",
        results_base_dir=results_base_dir,
        accuracy_thresholds=[accuracy_threshold],
        n_trials=5,
        data_fracs=np.logspace(-3,0,8),
        baselines = [],
        performance_metric=performance_metric,
        n_workers=1,
        verbose=verbose,
    )

    
