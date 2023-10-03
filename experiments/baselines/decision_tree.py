from sklearn.tree import DecisionTreeClassifier
from .baselines import SupervisedExperimentBaseline


class DecisionTreeClassifierBaseline(SupervisedExperimentBaseline):
    def __init__(self, **dt_kwargs):
        """Implements a decision tree classifier baseline for
        a binary classification task

        :param dt_kwargs: Any keyword arguments that scikit-learn's
        DecisionTreeClassifier takes.
        """
        SupervisedExperimentBaseline.__init__(self, model_name="decision_tree")
        self.dt_kwargs = dt_kwargs

    def train(self, X, Y):
        """Instantiate a new model instance and train (fit)
        it to the training data, X,y
        :param X: features
        :type X: 2D np.ndarray
        :param y: labels
        :type y: 1D np.ndarray
        """
        self.trained_model = DecisionTreeClassifier(**self.dt_kwargs)
        self.trained_model.fit(X, Y)  # parent method
        return None

    def predict(self, theta, X):
        """Use the trained model to predict positive class probabilities
        theta isn't used here because there are no fitted parameters
        in decision trees.
        :param theta: Model weights, None in this case
        :param X: features
        :type X: 2D np.ndarray
        """
        probs_bothclasses = self.trained_model.predict_proba(X)
        probs_posclass = probs_bothclasses[:, 1]
        return probs_posclass
