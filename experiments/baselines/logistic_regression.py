from seldonian.models.models import BinaryLogisticRegressionModel
from .baselines import SupervisedExperimentBaseline


class BinaryLogisticRegressionBaseline(
    BinaryLogisticRegressionModel, SupervisedExperimentBaseline
):
    def __init__(
        self,
    ):
        """Implements a logistic regression classifier for binary classification"""
        BinaryLogisticRegressionModel.__init__(
            self
        )  # inherits parent's predict() method.
        SupervisedExperimentBaseline.__init__(self, model_name="logistic_regression")

    def train(self, X, y):
        """Train the model. Just a wrapper to parent's fit() method.

        :param X: features
        :type X: 2D np.ndarray
        :param y: labels
        :type y: 1D np.ndarray
        """
        return self.fit(X, y)  # parent method
