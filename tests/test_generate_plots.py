import os
import numpy as np
import pandas as pd

import pytest

from experiments.generate_plots import (
    SupervisedPlotGenerator,RLPlotGenerator,
    CustomPlotGenerator)

from experiments.experiment_utils import (
    generate_episodes_and_calc_J,has_failed)

from experiments.perf_eval_funcs import (MSE,probabilistic_accuracy)

from seldonian.RL.environments.gridworld import Gridworld

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_regression_plot_generator(gpa_regression_spec,experiment):
    np.random.seed(42)
    constraint_strs = ['Mean_Squared_Error - 3.0','2.0 - Mean_Squared_Error']
    deltas = [0.05,0.1]
    spec = gpa_regression_spec(constraint_strs,deltas)
    n_trials = 2
    data_fracs = [0.01,0.1]
    datagen_method="resample"
    perf_eval_fn = MSE
    results_dir = "./tests/static/results"
    n_workers = 1
    # Get performance evaluation kwargs set up
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset

    test_features = dataset.features
    test_labels = dataset.labels

    # Define any additional keyword arguments (besides theta)
    # of the performance evaluation function,
    # which in our case is accuracy
    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        }
    
    spg = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        datagen_method=datagen_method,
        perf_eval_fn=perf_eval_fn,
        results_dir=results_dir,
        n_workers=n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs=perf_eval_kwargs,
        constraint_eval_kwargs={})
    
    assert spg.n_trials == n_trials
    assert spg.regime == 'supervised_learning'

    # Seldonian experiment

    spg.run_seldonian_experiment(verbose=True)

    ## Make sure results file was created
    results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
    assert os.path.exists(results_file)

    # Make sure length of df is correct
    df = pd.read_csv(results_file)
    assert len(df) == 4
    dps = df.data_frac
    trial_is = df.trial_i
    perfs = df.performance
    passed_safetys = df.passed_safety
    gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
    
    assert dps[0] == 0.01
    assert trial_is[0] == 0
    assert passed_safetys[0] == False
    assert gvecs.str[0][0] == -np.inf

    assert dps[1] == 0.01
    assert trial_is[1] == 1
    assert passed_safetys[1] == False
    assert gvecs.str[0][1] == -np.inf

    assert dps[2] == 0.1
    assert trial_is[2] == 0
    assert passed_safetys[2] == False
    assert gvecs.str[0][2] == -np.inf

    assert dps[3] == 0.1
    assert trial_is[3] == 1
    assert passed_safetys[3] == False
    assert gvecs.str[0][3] == -np.inf
    
    # Make sure number of trial files created is correct
    trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
    trial_files = os.listdir(trial_dir)
    assert len(trial_files) == 4

    # Make sure the trial files have the right format
    trial_file_0 = os.path.join(trial_dir,trial_files[0])
    df_trial0 = pd.read_csv(trial_file_0)
    assert len(df_trial0) == 1

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_custom_regime_plot_generator(custom_text_spec,experiment):
    np.random.seed(42)
    
    spec = custom_text_spec()
    n_trials = 2
    data_fracs = [0.1,0.5]
    datagen_method="resample"
    verbose=True
    def calc_performance(theta,model,data,**kwargs):
        # A dummy function
        y_pred = model.predict(theta,data)
        return np.mean(y_pred)

    perf_eval_fn = calc_performance
    results_dir = "./tests/static/results"
    n_workers = 1
    # Get performance evaluation kwargs set up
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset

    test_data = dataset.data

    # Define any additional keyword arguments (besides theta)
    # of the performance evaluation function,
    # which in our case is accuracy
    perf_eval_kwargs = {
        'test_data':test_data,
        }
    
    spg = CustomPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        datagen_method=datagen_method,
        perf_eval_fn=perf_eval_fn,
        results_dir=results_dir,
        n_workers=n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs=perf_eval_kwargs,
        constraint_eval_kwargs={})
    
    assert spg.n_trials == n_trials
    assert spg.regime == 'custom'

    # Generate resampled datasets
    spg.generate_resampled_datasets(verbose=verbose)

    # Make sure n_trials resampled datasets were created
    resampled_dir = os.path.join(results_dir,"resampled_datasets")
    resampled_files = os.listdir(resampled_dir)
    assert len(resampled_files) == n_trials
    assert "trial_0.pkl" in resampled_files and "trial_1.pkl" in resampled_files

    # Seldonian experiment

    spg.run_seldonian_experiment(verbose=True)

    # ## Make sure results file was created
    results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
    assert os.path.exists(results_file)

    # # Make sure length of df is correct
    df = pd.read_csv(results_file)
    assert len(df) == 4
    dps = df.data_frac
    trial_is = df.trial_i
    perfs = df.performance
    passed_safetys = df.passed_safety
    gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
    
    assert dps[0] == 0.1
    assert trial_is[0] == 0
    assert passed_safetys[0] == False
    assert gvecs.str[0][0] == -np.inf

    assert dps[1] == 0.1
    assert trial_is[1] == 1
    assert passed_safetys[1] == False
    assert gvecs.str[0][1] == -np.inf

    assert dps[2] == 0.5
    assert trial_is[2] == 0
    assert passed_safetys[2] == True
    assert -np.inf < gvecs.str[0][2] < 0 

    assert dps[3] == 0.5
    assert trial_is[3] == 1
    assert passed_safetys[3] == True
    assert -np.inf < gvecs.str[0][3] < 0
    
    # # Make sure number of trial files created is correct
    trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
    trial_files = os.listdir(trial_dir)
    assert len(trial_files) == 4

    # Make sure the trial files have the right format
    trial_file_0 = os.path.join(trial_dir,trial_files[0])
    df_trial0 = pd.read_csv(trial_file_0)
    assert len(df_trial0) == 1

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_custom_regime_addl_datasets_plot_generator(custom_text_addl_datasets_spec,experiment):
    np.random.seed(42)
    
    spec = custom_text_addl_datasets_spec()
    n_trials = 2
    data_fracs = [0.1,0.5]
    datagen_method="resample"
    verbose=True
    def calc_performance(theta,model,data,**kwargs):
        # A dummy function
        y_pred = model.predict(theta,data)
        return np.mean(y_pred)

    perf_eval_fn = calc_performance
    results_dir = "./tests/static/results"
    n_workers = 1
    # Get performance evaluation kwargs set up
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset

    test_data = dataset.data

    # Define any additional keyword arguments (besides theta)
    # of the performance evaluation function,
    # which in our case is accuracy
    perf_eval_kwargs = {
        'test_data':test_data,
    }
    # Define ground truth for additional dataset
    constraint_eval_kwargs = {
        "additional_datasets": spec.additional_datasets
    }

    
    spg = CustomPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        datagen_method=datagen_method,
        perf_eval_fn=perf_eval_fn,
        results_dir=results_dir,
        n_workers=n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs=perf_eval_kwargs,
        constraint_eval_kwargs=constraint_eval_kwargs
    )
    
    assert spg.n_trials == n_trials
    assert spg.regime == 'custom'

    # Generate resampled datasets
    spg.generate_resampled_datasets(verbose=verbose)

    # Make sure n_trials resampled datasets were created
    resampled_dir = os.path.join(results_dir,"resampled_datasets")
    resampled_files = os.listdir(resampled_dir)
    assert len(resampled_files) == n_trials*2
    assert "trial_0.pkl" in resampled_files
    assert "trial_1.pkl" in resampled_files
    assert "trial_0_addl_datasets.pkl" in resampled_files
    assert "trial_1_addl_datasets.pkl" in resampled_files

    # # Seldonian experiment

    spg.run_seldonian_experiment(verbose=True)

    # # ## Make sure results file was created
    results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
    assert os.path.exists(results_file)

    # # # Make sure length of df is correct
    df = pd.read_csv(results_file)
    assert len(df) == 4
    print(df)
    dps = df.data_frac
    trial_is = df.trial_i
    perfs = df.performance
    passed_safetys = df.passed_safety
    gvecs = df.gvec.apply(lambda t: np.fromstring(t[1:-1],sep=' '))
    
    assert dps[0] == 0.1
    assert trial_is[0] == 0
    assert passed_safetys[0] == False
    assert gvecs.str[0][0] == -np.inf

    assert dps[1] == 0.1
    assert trial_is[1] == 1
    assert passed_safetys[1] == False
    assert gvecs.str[0][1] == -np.inf

    assert dps[2] == 0.5
    assert trial_is[2] == 0
    assert passed_safetys[2] == True
    assert -np.inf < gvecs.str[0][2] < 0 

    assert dps[3] == 0.5
    assert trial_is[3] == 1
    assert passed_safetys[3] == True
    assert -np.inf < gvecs.str[0][3] < 0
    
    # Make sure number of trial files created is correct
    trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
    trial_files = os.listdir(trial_dir)
    assert len(trial_files) == 4

    # Make sure the trial files have the right format
    trial_file_0 = os.path.join(trial_dir,trial_files[0])
    df_trial0 = pd.read_csv(trial_file_0)
    assert len(df_trial0) == 1

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_bad_datagen_method(gpa_regression_spec,experiment):
    np.random.seed(42)
    constraint_strs = ['Mean_Squared_Error - 3.0','2.0 - Mean_Squared_Error']
    deltas = [0.05,0.1]
    spec = gpa_regression_spec(constraint_strs,deltas)
    n_trials = 2
    data_fracs = [0.01,0.1]
    datagen_method="bad_method"
    perf_eval_fn = MSE
    results_dir = "./tests/static/results"
    n_workers = 1
    # Get performance evaluation kwargs set up
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset

    test_features = dataset.features
    test_labels = dataset.labels

    # Define any additional keyword arguments (besides theta)
    # of the performance evaluation function,
    # which in our case is accuracy
    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        }
    
    spg = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        datagen_method=datagen_method,
        perf_eval_fn=perf_eval_fn,
        results_dir=results_dir,
        n_workers=n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs=perf_eval_kwargs,
        constraint_eval_kwargs={})
    
    assert spg.n_trials == n_trials
    assert spg.regime == 'supervised_learning'

    # Seldonian experiment
    with pytest.raises(NotImplementedError) as excinfo:
        spg.run_seldonian_experiment(verbose=True)

    error_str = "datagen_method: bad_method not supported for supervised learning."
    assert str(excinfo.value)

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_too_few_datapoints(gpa_regression_spec,experiment):
    """ Test that too small of a data_frac resulting in < 1
    data points in a trial raises an error """
    np.random.seed(42)
    constraint_strs = ['Mean_Squared_Error <= 2.0']
    deltas = [0.05]
    spec = gpa_regression_spec(constraint_strs,deltas)
    n_trials = 1
    data_fracs = [0.000001]
    datagen_method="resample"
    perf_eval_fn = MSE
    results_dir = "./tests/static/results"
    n_workers = 1
    # Get performance evaluation kwargs set up
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset

    test_features = dataset.features
    test_labels = dataset.labels

    # Define any additional keyword arguments (besides theta)
    # of the performance evaluation function,
    # which in our case is accuracy
    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        }
    
    spg = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        datagen_method=datagen_method,
        perf_eval_fn=perf_eval_fn,
        results_dir=results_dir,
        n_workers=n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs=perf_eval_kwargs,
        constraint_eval_kwargs={})
    
    assert spg.n_trials == n_trials
    assert spg.regime == 'supervised_learning'

    with pytest.raises(ValueError) as excinfo:
        spg.run_seldonian_experiment(verbose=True)
    error_str = (
        f"This data_frac={data_fracs[0]} "
        f"results in 0 data points. "
         "Must have at least 1 data point to run a trial.")

    assert str(excinfo.value) == error_str

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_addl_datasets_without_hold_out(gpa_regression_addl_datasets_spec,experiment):
    """ Test that if a specfile is used that contains addl datasets,
    but one does not provide addl datasets as hold out test sets, 
    an error is raised. """
    np.random.seed(42)
    constraint_strs = ['Mean_Squared_Error <= 2.0']
    deltas = [0.05]
    spec = gpa_regression_addl_datasets_spec(constraint_strs,deltas)
    n_trials = 1
    data_fracs = [0.01]
    datagen_method="resample"
    perf_eval_fn = MSE
    results_dir = "./tests/static/results"
    n_workers = 1
    # Get performance evaluation kwargs set up
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset
    test_features = dataset.features
    test_labels = dataset.labels

    # Define any additional keyword arguments (besides theta)
    # of the performance evaluation function,
    # which in our case is accuracy
    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        }

    # First completely omit addiitonal_datasets from constraint_eval_kwargs
    with pytest.raises(RuntimeError) as excinfo:
        spg = SupervisedPlotGenerator(
            spec=spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            results_dir=results_dir,
            n_workers=n_workers,
            constraint_eval_fns=[],
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs={})
    
    error_str = (
        "The 'additional_datasets' kwarg in 'constraint_eval_kwargs' is missing. "
        "This is needed because the spec object used to run this experiment "
        "contains additional datasets, and for each of those datasets, "
        "you need to provide a ground truth dataset so that the safety plot (right plot) can be evaluated."
    )

    assert str(excinfo.value) == error_str

    # Now include it but omit correct constraint str 
    constraint_eval_kwargs = {}
    constraint_eval_kwargs["additional_datasets"] = {"bad_string":{}}
    with pytest.raises(RuntimeError) as excinfo:
        spg = SupervisedPlotGenerator(
            spec=spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            results_dir=results_dir,
            n_workers=n_workers,
            constraint_eval_fns=[],
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs)
    error_str = "Constraint: 'Mean_Squared_Error-(2.0)' is missing from held out additional datasets."

    assert error_str == str(excinfo.value) 

    # Now include constraint str but omit correct base node
    constraint_eval_kwargs = {}
    constraint_eval_kwargs["additional_datasets"] = {
        "Mean_Squared_Error-(2.0)":{
            "bad_base_node": {}
        }
    }
    with pytest.raises(RuntimeError) as excinfo:
        spg = SupervisedPlotGenerator(
            spec=spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            results_dir=results_dir,
            n_workers=n_workers,
            constraint_eval_fns=[],
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs)
    error_str = (
        f"Base node: 'Mean_Squared_Error' in parse tree: 'Mean_Squared_Error-(2.0)' is missing "
        "from held out additional datasets dict."
    )

    assert error_str == str(excinfo.value) 

    # Now include constraint str and base node but exclude 'dataset' key
    constraint_eval_kwargs = {}
    constraint_eval_kwargs["additional_datasets"] = {
        "Mean_Squared_Error-(2.0)":{
            "Mean_Squared_Error": {"shmataset":[]}
        }
    }
    with pytest.raises(RuntimeError) as excinfo:
        spg = SupervisedPlotGenerator(
            spec=spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            results_dir=results_dir,
            n_workers=n_workers,
            constraint_eval_fns=[],
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs)
    error_str = (
        f"'dataset' key missing from held out additional datasets dict entry for "
        f"parse tree: 'Mean_Squared_Error-(2.0)' and base node 'Mean_Squared_Error'"
    )

    assert error_str == str(excinfo.value) 

    # Now include the 'dataset' key but the value is not a Seldonian DataSet
    constraint_eval_kwargs = {}
    constraint_eval_kwargs["additional_datasets"] = {
        "Mean_Squared_Error-(2.0)":{
            "Mean_Squared_Error": {"dataset":[1,2,3]}
        }
    }
    with pytest.raises(RuntimeError) as excinfo:
        spg = SupervisedPlotGenerator(
            spec=spec,
            n_trials=n_trials,
            data_fracs=data_fracs,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            results_dir=results_dir,
            n_workers=n_workers,
            constraint_eval_fns=[],
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs)
    error_str = (
        f"The dataset provided for parse tree: 'Mean_Squared_Error-(2.0)' and base node 'Mean_Squared_Error' "
        "is not a seldonian.DataSet object."
    )

    assert error_str == str(excinfo.value) 

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_addl_datasets_with_hold_out(gpa_regression_addl_datasets_spec,experiment):
    """ Test that when running an experiment with addl datasets,
    the held out addl datasets are used for safety evaluation"""
    np.random.seed(42)
    constraint_strs = ['Mean_Squared_Error <= 2.0']
    deltas = [0.05]
    spec = gpa_regression_addl_datasets_spec(constraint_strs,deltas)
    n_trials = 1
    data_fracs = [0.01]
    datagen_method="resample"
    perf_eval_fn = MSE
    results_dir = "./tests/static/results"
    n_workers = 1
    # Get performance evaluation kwargs set up
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset
    test_features = dataset.features
    test_labels = dataset.labels

    # Define any additional keyword arguments (besides theta)
    # of the performance evaluation function,
    # which in our case is accuracy
    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        }

    constraint_eval_kwargs = {}
    constraint_eval_kwargs["additional_datasets"] = spec.additional_datasets
    print()
    spg = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        datagen_method=datagen_method,
        perf_eval_fn=perf_eval_fn,
        results_dir=results_dir,
        n_workers=n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs=perf_eval_kwargs,
        constraint_eval_kwargs=constraint_eval_kwargs)

    assert spg.n_trials == n_trials
    assert spg.regime == 'supervised_learning'

    spg.run_seldonian_experiment(verbose=True)

    # Check that resampled primary and additional datasets files were created
    primary_resampled_file = os.path.join(results_dir,"resampled_datasets/trial_0.pkl")
    assert os.path.exists(primary_resampled_file)

    addl_resampled_file = os.path.join(results_dir,"resampled_datasets/trial_0_addl_datasets.pkl")
    assert os.path.exists(addl_resampled_file)

@pytest.mark.parametrize('experiment', ["./tests/static/gridworld_results"], indirect=True)
def test_too_few_episodes(gridworld_spec,experiment):
    """ Test that too small of a data_frac resulting in < 1
    episodes in a trial raises an error """
    np.random.seed(42)
    constraint_strs = ['J_pi_new_IS >= -0.25']
    deltas=[0.05]
    spec = gridworld_spec(constraint_strs,deltas)
    n_trials = 1
    data_fracs = [0.000001]
    datagen_method="generate_episodes"
    perf_eval_fn = generate_episodes_and_calc_J
    results_dir = "./tests/static/gridworld_results"
    n_workers = 1
    # Get performance evaluation kwargs set up
    n_episodes_for_eval = 100
    perf_eval_kwargs = {'n_episodes_for_eval':n_episodes_for_eval}
    
    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["env"] = Gridworld()
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparameter_and_setting_dict["num_episodes"] = 100
    hyperparameter_and_setting_dict["num_trials"] = 1
    hyperparameter_and_setting_dict["vis"] = False

    spg = RLPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        datagen_method=datagen_method,
        hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
        perf_eval_fn=perf_eval_fn,
        results_dir=results_dir,
        n_workers=n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs=perf_eval_kwargs,
        constraint_eval_kwargs={})
    
    assert spg.n_trials == n_trials
    assert spg.regime == 'reinforcement_learning'

    with pytest.raises(ValueError) as excinfo:
        spg.run_seldonian_experiment(verbose=True)
    error_str = (
        f"This data_frac={data_fracs[0]} "
        f"results in 0 episodes. "
         "Must have at least 1 episode to run a trial.")

    assert str(excinfo.value) == error_str

@pytest.mark.parametrize('experiment', ["./tests/static/results"], indirect=True)
def test_RL_plot_generator(gridworld_spec,experiment):
    np.random.seed(42)
    constraint_strs = ['J_pi_new_IS >= - 0.25']
    deltas = [0.05]
    spec = gridworld_spec(constraint_strs,deltas)
    spec.optimization_hyperparams['num_iters'] = 10
    n_trials = 2
    data_fracs = [0.05,0.1]
    datagen_method="generate_episodes"
    perf_eval_fn = generate_episodes_and_calc_J
    results_dir = "./tests/static/results"
    n_workers = 1
    n_episodes_for_eval=100
    # Get performance evaluation kwargs set up
    # Use entire original dataset as ground truth for test set
    dataset = spec.dataset
    
    # Define any additional keyword arguments (besides theta)
    # of the performance evaluation function,
    perf_eval_kwargs = {
        'n_episodes_for_eval':n_episodes_for_eval
    }
    
    hyperparameter_and_setting_dict = {}
    hyperparameter_and_setting_dict["env"] = Gridworld()
    hyperparameter_and_setting_dict["agent"] = "Parameterized_non_learning_softmax_agent"
    hyperparameter_and_setting_dict["num_episodes"] = 100
    hyperparameter_and_setting_dict["num_trials"] = 1
    hyperparameter_and_setting_dict["vis"] = False

    spg = RLPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        datagen_method=datagen_method,
        hyperparameter_and_setting_dict=hyperparameter_and_setting_dict,
        perf_eval_fn=perf_eval_fn,
        results_dir=results_dir,
        n_workers=n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs=perf_eval_kwargs,
        constraint_eval_kwargs={})
    
    assert spg.n_trials == n_trials
    assert spg.regime == 'reinforcement_learning'

    # Seldonian experiment

    spg.run_seldonian_experiment(verbose=True)

    ## Make sure results file was created
    results_file = os.path.join(results_dir,"qsa_results/qsa_results.csv")
    assert os.path.exists(results_file)

    # Make sure length of df is correct
    df = pd.read_csv(results_file)
    assert len(df) == 4
    dps = df.data_frac
    trial_is = df.trial_i
    perfs = df.performance
    passed_safetys = df.passed_safety
    print("df:")
    print(df)
    
    assert dps[0] == 0.05
    assert trial_is[0] == 0

    assert dps[1] == 0.05
    assert trial_is[1] == 1

    assert dps[2] == 0.1
    assert trial_is[2] == 0

    assert dps[3] == 0.1
    assert trial_is[3] == 1
    
    # Make sure number of trial files created is correct
    trial_dir = os.path.join(results_dir,"qsa_results/trial_data")
    trial_files = os.listdir(trial_dir)
    assert len(trial_files) == 4

    # Make sure the trial files have the right format
    trial_file_0 = os.path.join(trial_dir,trial_files[0])
    df_trial0 = pd.read_csv(trial_file_0)
    assert len(df_trial0) == 1

    # Now make plot
    savename = os.path.join(results_dir,"test_gridworld_plot.png")
    spg.make_plots(fontsize=12,legend_fontsize=8,
        performance_label='-IS_estimate',
        save_format="png",
        savename=savename)
    # Make sure it was saved
    assert os.path.exists(savename)
