import autograd.numpy as np
from seldonian.models.models import BinaryLogisticRegressionModel


class BinaryLogisticRegressionBaseline(BinaryLogisticRegressionModel):
    def __init__(self,):
        """Implements a classifier that always predicts
        that the positive class has prob=0.5,
        regardless of input"""
        super().__init__()
        self.model_name = "logistic_regression"

    def train(self,X,Y):
        return self.fit(X,Y) # parent method

