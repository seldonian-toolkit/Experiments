## Run headless mnist experiments (not yet documented on the website)

### Epsilon is the accuracy threshold:
# 

import argparse
import numpy as np
import math
import os
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.headless_example import HeadlessExample
from experiments.experiment_utils import (
    make_batch_epoch_dict_min_sample_repeat)
from experiments.perf_eval_funcs import (probabilistic_accuracy, binary_logistic_loss)
from experiments.headless_utils import make_data_loaders
from seldonian.utils.io_utils import load_pickle
from examples.headless_facial_gender.full_model import CNN
from examples.headless_facial_gender.headless_model import CNN_headless

from experiments.baselines.facial_recog_cnn import PytorchFacialRecogBaseline

import torch
import torch.nn as nn

def headless_facial_gender_example(
    spec_rootdir,
    results_base_dir,
    epsilons=[0.8],
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
    head_layer_names = ['fc3.weight','fc3.bias']
    
    data_fracs = np.array(data_fracs)
    # Ensure that number of iterations in pretraining
    # is max(niter_min_pretraining,# of iterations s.t. each sample is seen num_repeats times)
    niter_min_pretraining=25
    num_repeats=4
    batch_size_pretraining=100
    N_candidate_max=11850
    batch_epoch_dict_pretraining = make_batch_epoch_dict_min_sample_repeat(
        niter_min_pretraining,
        data_fracs,
        N_candidate_max,
        batch_size_pretraining,
        num_repeats)

    # We're not batching in candidate selection because the head-only 
    # optimization problem is small enough.
    
    batch_epoch_dict = {}
    
    # The function that will calculate performance
    if performance_metric == "accuracy":
        perf_eval_fn = probabilistic_accuracy
        performance_yscale='linear'
    elif performance_metric == 'log_loss':
        perf_eval_fn = binary_logistic_loss
        performance_yscale='log'
    else:
        raise NotImplementedError(
            f"Performance metric {performance_metric} not supported for this example")

    for epsilon in epsilons:
        specfile = os.path.join(
            spec_rootdir,
            f"headless_facial_gender_overall_accuracy_equality_{epsilon}.pkl"
        )
        spec = load_pickle(specfile)
        results_dir = os.path.join(results_base_dir,
            f"headless_facial_gender_overall_accuracy_equality_{epsilon}_perf_{performance_metric}")
        plot_savename = os.path.join(
            results_dir, f"headless_facial_gender_overall_accuracy_equality_{epsilon}_perf_{performance_metric}.pdf"
        )
        # Set up kwargs that will be passed into the perf_eval_fn function
        dataset = spec.dataset
        test_features = dataset.features
        test_labels = dataset.labels

        # Make data loaders out of features and labels
        # The batch sizes here are only used when passing 
        # the data through the network a single time to create the latent features, 
        # Batching is only done for memory concerns, not for training purposes
        test_data_loaders = make_data_loaders(
            test_features, 
            test_labels, 
            spec.frac_data_in_safety, 
            candidate_batch_size=1000, 
            safety_batch_size=1000,
        )

        perf_eval_kwargs = {
            "test_data_loaders": test_data_loaders,
            "X":test_features,
            "y": test_labels,
            "eval_batch_size": 2000, # how many latent features are passed through perf_eval_fn at a time. Lower this if running into memory issues
        }
        ex = HeadlessExample(spec=spec)

        ex.run(
            full_pretraining_model=full_pretraining_model,
            headless_pretraining_model=headless_pretraining_model,
            head_layer_names=head_layer_names,
            latent_feature_shape=(256,),
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
    # parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('--accuracy_threshold', help='Accuracy >= accuracy_threshold', required=True)
    # parser.add_argument('--n_trials', help='Number of trials to run', required=True)
    # parser.add_argument('--n_workers', help='Number of workers to use', required=True)
    # parser.add_argument('--include_baselines', help='verbose', action="store_true")
    # parser.add_argument('--verbose', help='verbose', action="store_true")

    # args = parser.parse_args()

    # accuracy_threshold = float(args.accuracy_threshold)
    # n_trials = int(args.n_trials)
    # n_workers = int(args.n_workers)
    # include_baselines = args.include_baselines
    # verbose = args.verbose

    # if include_baselines:
    #     baselines = ["random_classifier","logistic_regression"]
    # else:
    #     baselines = []
    data_fracs = np.array([0.05,0.2,0.85])

    niter_min_baseline=25 # how many iterations we want in each run. Overfitting happens with more than this.
    N_candidate_max=11850
    batch_size_baseline=100
    num_repeats=4
    batch_epoch_dict_baseline = make_batch_epoch_dict_min_sample_repeat(
        niter_min_baseline,
        data_fracs,
        N_candidate_max,
        batch_size_baseline,
        num_repeats)
    print("batch_epoch_dict_baseline:")
    print(batch_epoch_dict_baseline)
    facial_recog_baseline = PytorchFacialRecogBaseline(
        device=torch.device('mps'),
        learning_rate = 0.001,
        batch_epoch_dict=batch_epoch_dict_baseline
        )
    epsilon = 0.8
    performance_metric="accuracy"

    # results_base_dir = f"./results"
    results_base_dir = f"./test_results"

    headless_facial_gender_example(
        spec_rootdir="data/spec/",
        results_base_dir=results_base_dir,
        epsilons=[epsilon],
        # n_trials=20,
        n_trials=2,
        # data_fracs=np.array([0.0005,0.005,0.05,0.075,0.1,0.25,0.5,0.75,1.0]),
        data_fracs=data_fracs,
        baselines = [facial_recog_baseline],
        performance_metric=performance_metric,
        n_workers=1,
        verbose=True,
    )

    
