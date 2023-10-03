import autograd.numpy as np
from seldonian.models.models import ClassificationModel, RandomClassifierModel
from .baselines import SupervisedExperimentBaseline


class UniformRandomClassifierBaseline(
    RandomClassifierModel, SupervisedExperimentBaseline
):
    def __init__(
        self,
    ):
        """Implements a classifier that always predicts
        that the positive class has prob=0.5,
        regardless of input"""
        RandomClassifierModel.__init__(self)
        SupervisedExperimentBaseline.__init__(self, model_name="uniform_random")

    def train(self, X, Y):
        return None


class WeightedRandomClassifierBaseline(
    RandomClassifierModel, SupervisedExperimentBaseline
):
    def __init__(self, weight):
        """Implements a classifier that always predicts
        that the positive class has prob=0.5,
        regardless of input"""
        RandomClassifierModel.__init__(self)
        SupervisedExperimentBaseline.__init__(
            self, model_name="weighted_random_classifier"
        )
        assert 0.0 <= weight <= 1.0
        self.weight = weight
        self.model_name = f"weighted_random_{weight:.2f}"

    def train(self, X, Y):
        return None

    def predict(self, theta, X):
        """Overrides parent method. Predict the probability of
        having the positive class label

        :param theta: The parameter weights
        :type theta: numpy ndarray
        :param X: The features
        :type X: numpy ndarray
        :return: predictions for each observation
        :rtype: float
        """
        return self.weight * np.ones(len(X))
