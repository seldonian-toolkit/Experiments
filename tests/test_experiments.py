import pytest

from experiments.experiments import (
	BaselineExperiment,SeldonianExperiment)

from experiments.utils import MSE

def test_create_seldonian_experiment():
	sd_exp = SeldonianExperiment(model_name='qsa',results_dir="./results")
	assert sd_exp.model_name == 'qsa'
	with pytest.raises(NotImplementedError) as excinfo:
		sd_exp_badname = SeldonianExperiment(model_name='SA',results_dir="./results")
	error_str = (
		"Seldonian experiments for model: "
		"SA are not supported.")

	assert str(excinfo.value) == error_str

def test_create_baseline_experiment():
	bl_exp = BaselineExperiment(model_name='logistic_regression',results_dir="./results")
	assert bl_exp.model_name == 'logistic_regression'


	
