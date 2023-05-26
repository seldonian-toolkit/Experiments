import pytest

from experiments.experiments import (
	BaselineExperiment,SeldonianExperiment)

from experiments.perf_eval_funcs import MSE
from experiments.baselines.logistic_regression import BinaryLogisticRegressionBaseline

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
	bl_model = BinaryLogisticRegressionBaseline()
	bl_exp = BaselineExperiment(baseline_model=bl_model,results_dir="./results")
	assert bl_exp.model_name == 'logistic_regression'


	
