# createSpec.py
import autograd.numpy as np
import os
from seldonian.parse_tree.parse_tree import make_parse_trees_from_constraints
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (load_json,save_pickle)
from seldonian.spec import SupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel)
from seldonian.models import objectives



if __name__ == '__main__':

    def initial_solution_fn(model,X,Y):
        return model.fit(X,Y)
    data_pth = "../../static/datasets/supervised/GPA/gpa_classification_dataset.csv"
    metadata_pth = "../../static/datasets/supervised/GPA/metadata_classification.json"
    save_base_dir = '.'
    # save_base_dir='.'
    # Load metadata
    regime='supervised_learning'
    sub_regime='classification'
    frac_data_in_safety = 0.6
    verbose=True

    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')
    
    # Behavioral constraints
    deltas = [0.05]
    model = BinaryLogisticRegressionModel()
    
    for constraint_name in ["disparate_impact",
        "demographic_parity","equalized_odds",
        "equal_opportunity","predictive_equality"]:
        save_dir = os.path.join(save_base_dir,f'gpa_{constraint_name}')
        os.makedirs(save_dir,exist_ok=True)
        # Define behavioral constraints
        if constraint_name == 'disparate_impact':
            constraint_strs = ['0.8 - min((PR | [M])/(PR | [F]),(PR | [F])/(PR | [M]))'] 
        elif constraint_name == 'demographic_parity':
            constraint_strs = ['abs((PR | [M]) - (PR | [F])) <= 0.2']
        elif constraint_name == 'equalized_odds':
            constraint_strs = ['abs((FNR | [M]) - (FNR | [F])) + abs((FPR | [M]) - (FPR | [F])) <= 0.35']
        elif constraint_name == 'equal_opportunity':
            constraint_strs = ['abs((FNR | [M]) - (FNR | [F])) <= 0.2']
        elif constraint_name == 'predictive_equality':
            constraint_strs = ['abs((FPR | [M]) - (FPR | [F])) <= 0.2']

        parse_trees = make_parse_trees_from_constraints(
            constraint_strs,
            deltas,
            regime="supervised_learning",
            sub_regime=sub_regime,
            columns=dataset.meta.sensitive_col_names,
            delta_weight_method="equal",
        )

        spec = SupervisedSpec(
            dataset=dataset,
            model=model,
            frac_data_in_safety=frac_data_in_safety,
            primary_objective=objectives.binary_logistic_loss,
            use_builtin_primary_gradient_fn=True,
            parse_trees=parse_trees,
            sub_regime=sub_regime,
            initial_solution_fn=None,
            optimization_technique="gradient_descent",
            optimizer="adam",
            optimization_hyperparams={
                "lambda_init": np.array([0.5]),
                "alpha_theta": 0.01,
                "alpha_lamb": 0.01,
                "beta_velocity": 0.9,
                "beta_rmsprop": 0.95,
                "use_batches": False,
                "num_iters": 1000,
                "gradient_library": "autograd",
                "hyper_search": None,
                "verbose": verbose,
            },
            verbose=verbose,
        )

        spec_save_name = os.path.join(save_dir, "spec.pkl")
        save_pickle(spec_save_name, spec, verbose=verbose)