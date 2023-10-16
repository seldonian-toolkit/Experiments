from sklearn.ensemble import RandomForestClassifier
from .baselines import SupervisedExperimentBaseline


class RandomForestClassifierBaseline(SupervisedExperimentBaseline):
    def __init__(self, **rf_kwargs):
        """Implements a random forest classifier baseline for
        a binary classification task"""
        SupervisedExperimentBaseline.__init__(self, model_name="random_forest")
        self.rf_kwargs = rf_kwargs

    def train(self, X, Y):
        """Instantiate a new model instance and train (fit)
        it to the training data, X,y
        :param X: features
        :type X: 2D np.ndarray
        :param y: labels
        :type y: 1D np.ndarray
        """
        self.trained_model = RandomForestClassifier(**self.rf_kwargs)
        self.trained_model.fit(X, Y)  # parent method
        return

    def predict(self, theta, X):
        """Use the trained model to predict positive class probabilities
        theta isn't used here because there are no fitted parameters
        in random forests.
        :param theta: Model weights, None in this case
        :param X: features
        :type X: 2D np.ndarray
        """
        probs_bothclasses = self.trained_model.predict_proba(X)
        probs_posclass = probs_bothclasses[:, 1]
        return probs_posclass
