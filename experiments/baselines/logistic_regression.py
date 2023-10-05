from seldonian.models.models import BinaryLogisticRegressionModel


class BinaryLogisticRegressionBaseline(BinaryLogisticRegressionModel):
    def __init__(self,):
        """Implements a logistic regression classifier for binary classification"""
        super().__init__()
        self.model_name = "logistic_regression"

    def train(self,X,y):
        """Train the model. Just a wrapper to parent fit() method.

        :param X: features
        :type X: 2D np.ndarray 
        :param y: labels
        :type y: 1D np.ndarray
        """
        return self.fit(X,y) # parent method

