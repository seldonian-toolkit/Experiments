import pytest

from experiments.experiments import (
	BaselineExperiment,SeldonianExperiment)

def test_create_seldonian_experiment():
	sd_exp = SeldonianExperiment(model_name='QSA',results_dir="./results")
	assert sd_exp.model_name == 'QSA'
	with pytest.raises(NotImplementedError) as excinfo:
		sd_exp_badname = SeldonianExperiment(model_name='SA',results_dir="./results")
	error_str = (
		"Seldonian experiments for model: "
		"SA are not supported.")

	assert str(excinfo.value) == error_str
