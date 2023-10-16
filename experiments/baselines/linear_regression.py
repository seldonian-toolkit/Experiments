import autograd.numpy as np
from seldonian.models.models import LinearRegressionModel
from .baselines import SupervisedExperimentBaseline


class LinearRegressionBaseline(LinearRegressionModel, SupervisedExperimentBaseline):
    def __init__(self):
        """Implements a classifier that always predicts
        that the positive class has prob=0.5,
        regardless of input"""
        LinearRegressionModel.__init__(self)  # inherits parent's predict() method.
        SupervisedExperimentBaseline.__init__(self, model_name="linear_regression")

    def train(self, X, Y):
        """Train the model. Just a wrapper to parent's fit() method.

        :param X: features
        :type X: 2D np.ndarray
        :param y: labels
        :type y: 1D np.ndarray
        """
        return self.fit(X, Y)
