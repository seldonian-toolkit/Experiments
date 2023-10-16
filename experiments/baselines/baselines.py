import autograd.numpy as np


class SupervisedExperimentBaseline:
    def __init__(self, model_name):
        """Base class for all supervised learning experiment baselines. All such baselines
        must implement at least the two methods below: train() and predict()."""
        self.model_name = model_name

    def train(self, X, Y):
        """Train the model on the training data, X,y
        :param X: features
        :type X: 2D np.ndarray
        :param y: labels
        :type y: 1D np.ndarray
        """
        raise NotImplementedError("Implement this method in a child class")

    def predict(self, theta, X):
        """Use the trained model to make predictions on new features, X
        :param theta: Model weights, not always used.
        :param X: features
        :type X: 2D np.ndarray
        """
        raise NotImplementedError("Implement this method in a child class")


class RLExperimentBaseline(object):
    def __init__(self, model_name, policy, env_kwargs={"gamma": 1.0}):
        """Base class for all RL experiment baselines. All RL experiment baselines
        must have at least the two methods below. Depending on the constraint,
        other methods may be required. When the constraint involves an importance sampling variant,
        e.g., one of the "J_pi_new" variants, a method:
            get_probs_from_observations_and_actions(
                self,
                theta,
                observations,
                actions,
                behavior_action_probs
            )
        is also required.
        """
        self.model_name = model_name
        self.policy = policy
        self.env_kwargs = env_kwargs
        self.gamma = env_kwargs["gamma"]  # discount factor

    def set_new_params(self, weights):
        """Set new policy parameters given model weights"""
        raise NotImplementedError("Implement this method in a child class")

    def train(self, dataset, **kwargs):
        """
        Train the model using a Seldonian dataset object. This contains the episodes
        generated using the behavior policy. Must return the trained policy parameters.
        """
        raise NotImplementedError("Implement this method in a child class")
