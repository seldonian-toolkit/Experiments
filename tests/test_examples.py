import os
import numpy as np
import pandas as pd

import pytest

from experiments.base_example import BaseExample
from examples.gpa_science_classification.generate_experiment_plots import gpa_example
from examples.lie_detection.generate_experiment_plots import lie_detection_example
from experiments.baselines.logistic_regression import BinaryLogisticRegressionBaseline

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_gpa_example(experiment):
    np.random.seed(42)

    example_name = "gpa_science_classification_disparate_impact_0.8_accuracy"

    results_base_dir = "./tests/static/results"
    gpa_example(
        spec_rootdir="static/specfiles/GPA",
        results_base_dir=results_base_dir,
        constraints=["disparate_impact"],
        n_trials=2,
        data_fracs=[0.01,0.1],
        baselines=[BinaryLogisticRegressionBaseline()],
        include_fairlearn_models=True,
        performance_metric="accuracy",
        n_workers=1,
    )

    # QSA tests
    qsa_results_file = os.path.join(
        results_base_dir,
        example_name,
        "qsa_results","qsa_results.csv")
    assert os.path.exists(qsa_results_file)

    df = pd.read_csv(qsa_results_file)
    assert len(df) == 4
    dps = df.data_frac
    trial_is = df.trial_i
    perfs = df.performance
    passed_safetys = df.passed_safety
    gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
    
    assert dps[0] == 0.01
    assert trial_is[0] == 0
    assert perfs[0] > 0 or np.isnan(perfs[0])
    assert passed_safetys[0] in [True,False]
    assert gvecs[0][0] <= 0 or np.isnan(gvecs[0][0])

    assert dps[1] == 0.01
    assert trial_is[1] == 1
    assert perfs[1] > 0 or np.isnan(perfs[1])
    assert passed_safetys[1] in [True,False]
    assert gvecs[1][0] <= 0 or np.isnan(gvecs[1][0])

    assert dps[2] == 0.1
    assert trial_is[2] == 0
    assert perfs[2] > 0 or np.isnan(perfs[2])
    assert passed_safetys[2] in [True,False]
    assert gvecs[2][0] <= 0 or np.isnan(gvecs[2][0])

    assert dps[3] == 0.1
    assert trial_is[3] == 1
    assert perfs[3] > 0 or np.isnan(perfs[3])
    assert passed_safetys[3] in [True,False]
    assert gvecs[3][0] <= 0 or np.isnan(gvecs[3][0])    

    # Make sure number of trial files created is correct
    trial_dir = os.path.join(results_base_dir,example_name,"qsa_results/trial_data")
    trial_files = os.listdir(trial_dir)
    assert len(trial_files) == 4

    # LR baseline tests
    lr_results_file = os.path.join(
        results_base_dir,
        example_name,
        "logistic_regression_results","logistic_regression_results.csv")
    assert os.path.exists(lr_results_file)

    df = pd.read_csv(lr_results_file)
    assert len(df) == 4
    dps = df.data_frac
    trial_is = df.trial_i
    perfs = df.performance
    gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
    
    assert dps[0] == 0.01
    assert trial_is[0] == 0
    assert perfs[0] > 0 or np.isnan(perfs[0])
    assert gvecs[0][0] < 0 or np.isnan(gvecs[0][0])

    assert dps[1] == 0.01
    assert trial_is[1] == 1
    assert perfs[1] > 0 or np.isnan(perfs[1])
    assert gvecs[1][0] < 0 or np.isnan(gvecs[1][0])

    assert dps[2] == 0.1
    assert trial_is[2] == 0
    assert perfs[2] > 0 or np.isnan(perfs[2])
    assert gvecs[2][0] < 0 or np.isnan(gvecs[2][0])

    assert dps[3] == 0.1
    assert trial_is[3] == 1
    assert perfs[3] > 0 or np.isnan(perfs[3])
    assert gvecs[3][0] < 0 or np.isnan(gvecs[3][0])    

    # Make sure number of trial files created is correct
    trial_dir = os.path.join(results_base_dir,example_name,"logistic_regression_results/trial_data")
    trial_files = os.listdir(trial_dir)
    assert len(trial_files) == 4
    
    # Fairlearn tests
    fairlearn_constraint_epsilons = [0.01,0.1,1.0]
    for epsilon in fairlearn_constraint_epsilons:
        fairlearn_model_name = f"fairlearn_eps{epsilon:.2f}"
        fairlearn_results_file = os.path.join(
            results_base_dir,
            example_name,
            f"{fairlearn_model_name}_results",
            f"{fairlearn_model_name}_results.csv"
        )
        assert os.path.exists(fairlearn_results_file)

        df = pd.read_csv(fairlearn_results_file)
        assert len(df) == 4
        dps = df.data_frac
        trial_is = df.trial_i
        perfs = df.performance
        gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
        
        assert dps[0] == 0.01
        assert trial_is[0] == 0
        assert perfs[0] > 0 or np.isnan(perfs[0])
        assert gvecs[0][0] <= 0 or gvecs[0][0] > 0 or np.isnan(gvecs[0][0]) # checking that it is a number or nan

        assert dps[1] == 0.01
        assert trial_is[1] == 1
        assert perfs[1] > 0 or np.isnan(perfs[1])
        assert gvecs[1][0] <= 0 or gvecs[1][0] > 0 or np.isnan(gvecs[1][0]) # checking that it is a number or nan

        assert dps[2] == 0.1
        assert trial_is[2] == 0
        assert perfs[2] > 0 or np.isnan(perfs[2])
        assert gvecs[2][0] <= 0 or gvecs[2][0] > 0 or np.isnan(gvecs[2][0]) # checking that it is a number or nan

        assert dps[3] == 0.1
        assert trial_is[3] == 1
        assert perfs[3] > 0 or np.isnan(perfs[3])
        assert gvecs[3][0] <= 0 or gvecs[3][0] > 0 or np.isnan(gvecs[3][0]) # checking that it is a number or nan    

        # Make sure number of trial files created is correct
        trial_dir = os.path.join(results_base_dir,example_name,f"{fairlearn_model_name}_results/trial_data")
        trial_files = os.listdir(trial_dir)
        assert len(trial_files) == 4

    # Make sure figure was saved
    savename = os.path.join(
        results_base_dir,
        example_name,
        "disparate_impact_0.8_accuracy.pdf",
    )
    assert os.path.exists(savename)

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_lie_detection_example(experiment):
    np.random.seed(42)

    results_base_dir = "./tests/static/results"
    example_name="lie_detection_overall_accuracy_equality_0.2_accuracy"

    lie_detection_example(
        spec_rootdir="static/specfiles/lie_detection",
        results_base_dir=results_base_dir,
        constraints = [
            "overall_accuracy_equality",
        ],
        epsilons=[0.2],
        n_trials=2,
        data_fracs=[0.01,0.1],
        baselines = [BinaryLogisticRegressionBaseline()],
        performance_metric="accuracy",
        n_workers=1,
    )
    # QSA tests
    qsa_results_file = os.path.join(
        results_base_dir,
        example_name,
        "qsa_results","qsa_results.csv")
    assert os.path.exists(qsa_results_file)

    df = pd.read_csv(qsa_results_file)
    assert len(df) == 4
    dps = df.data_frac
    trial_is = df.trial_i
    perfs = df.performance
    passed_safetys = df.passed_safety
    gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
    # Needs to have gvec < 0 (i.e. safe on test data) or nan, but not gvec > 0
    assert dps[0] == 0.01
    assert trial_is[0] == 0
    assert perfs[0] > 0 or np.isnan(perfs[0])
    assert passed_safetys[0] in [True,False]
    assert gvecs[0][0] < 0 or np.isnan(gvecs[0][0])

    assert dps[1] == 0.01
    assert trial_is[1] == 1
    assert perfs[1] > 0 or np.isnan(perfs[1])
    assert passed_safetys[1] in [True,False]
    assert gvecs[1][0] < 0 or np.isnan(gvecs[1][0])

    assert dps[2] == 0.1
    assert trial_is[2] == 0
    assert perfs[2] > 0 or np.isnan(perfs[2])
    assert passed_safetys[2] in [True,False]
    assert gvecs[2][0] < 0 or np.isnan(gvecs[2][0])

    assert dps[3] == 0.1
    assert trial_is[3] == 1
    assert perfs[3] > 0 or np.isnan(perfs[3])
    assert passed_safetys[3] in [True,False]
    assert gvecs[3][0] < 0 or np.isnan(gvecs[3][0])    

    # # Make sure number of trial files created is correct
    trial_dir = os.path.join(results_base_dir,example_name,"qsa_results/trial_data")
    trial_files = os.listdir(trial_dir)
    assert len(trial_files) == 4

    # LR baseline tests
    lr_results_file = os.path.join(
        results_base_dir,
        example_name,
        "logistic_regression_results","logistic_regression_results.csv")
    assert os.path.exists(lr_results_file)

    df = pd.read_csv(lr_results_file)
    assert len(df) == 4
    dps = df.data_frac
    trial_is = df.trial_i
    perfs = df.performance
    gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
    # LR could fail or pass, gvec just needs to be a number
    assert dps[0] == 0.01
    assert trial_is[0] == 0
    assert perfs[0] > 0 or np.isnan(perfs[0])
    assert gvecs[0][0] <= 0 or gvecs[0][0] > 0 or np.isnan(gvecs[0][0])

    assert dps[1] == 0.01
    assert trial_is[1] == 1
    assert perfs[1] > 0 or np.isnan(perfs[1])
    assert gvecs[1][0] <= 0 or gvecs[1][0] > 0 or np.isnan(gvecs[1][0])

    assert dps[2] == 0.1
    assert trial_is[2] == 0
    assert perfs[2] > 0 or np.isnan(perfs[2])
    assert gvecs[2][0] <= 0 or gvecs[2][0] > 0 or np.isnan(gvecs[2][0])

    assert dps[3] == 0.1
    assert trial_is[3] == 1
    assert perfs[3] > 0 or np.isnan(perfs[3])
    assert gvecs[3][0] <= 0 or gvecs[3][0] > 0 or np.isnan(gvecs[3][0])    

    # Make sure number of trial files created is correct
    trial_dir = os.path.join(results_base_dir,example_name,"logistic_regression_results/trial_data")
    trial_files = os.listdir(trial_dir)
    assert len(trial_files) == 4
    
    # Make sure figure was saved
    savename = os.path.join(
        results_base_dir,
        example_name,
        "overall_accuracy_equality_0.2_accuracy.pdf",
    )
    assert os.path.exists(savename)
