from sklearn.tree import DecisionTreeClassifier
from autograd import grad
import autograd.numpy as np
from seldonian.models.trees.sktree_model import SeldonianDecisionTree, probs2theta
from seldonian.warnings.custom_warnings import *
from .baselines import SupervisedExperimentBaseline


class DecisionTreeClassifierLeafTuningBaseline(
    SeldonianDecisionTree, SupervisedExperimentBaseline
):
    def __init__(self, primary_objective_fn, sub_regime, adam_kwargs, dt_kwargs={}):
        """Implements a decision classifier with leaf node tuning
        as a baseline for binary classification tasks

        :param primary_objective: The primary objective function to be minimized
            during the leaf tuning process.
        :param sub_regime: Sub category of ML problem, e.g., "classification"
        :param adam_kwargs: Keyword arguments to pass to the adam optimizer
        :param dt_kwargs: Any keyword arguments that scikit-learn's
            DecisionTreeClassifier takes.
        """
        SeldonianDecisionTree.__init__(self, **dt_kwargs)
        SupervisedExperimentBaseline.__init__(
            self, model_name="decision_tree_leaf_tuning"
        )
        self.primary_objective_fn = primary_objective_fn
        self.sub_regime = sub_regime
        self.adam_kwargs = adam_kwargs

    def train(self, X, Y):
        """Instantiate a new model instance and train (fit)
        it to the training data, X,Y. Then run Adam gradient descent
        on the leaf node probabilities.
        :param X: features
        :type X: 2D np.ndarray
        :param Y: labels
        :type Y: 1D np.ndarray
        """
        leaf_node_probs = self.fit(X, Y)  # parent method
        self.X = X
        self.Y = Y
        # Convert probs to theta so I can run gradient descent on them
        theta_init = probs2theta(leaf_node_probs)
        theta = self.adam(theta_init, **self.adam_kwargs)
        return theta

    def wrapped_primary_objective(self, theta):
        """The first argument of primary_objective_fn is model,
        which is used to get the predictions via .predict(). We
        want to use the current model's predict() method (inherited) for this,
        so we use self for first arg.

        :param theta: Model weights

        :return: The primary objective function evaluated with these theta weights
        """
        return self.primary_objective_fn(
            self, theta, self.X, self.Y, sub_regime=self.sub_regime
        )

    def adam(self, theta_init, **kwargs):
        """Perform gradient descent with adam optimizer

        :param theta_init: The initial weights to begin gradient descent

        :return: best_theta, the optimal weights that minimize the primary objective function
        """
        n_iters_tot = kwargs["n_iters_tot"]
        verbose = kwargs["verbose"]
        debug = kwargs["debug"]
        theta = theta_init
        # initialize Adam parameters
        alpha_theta = 0.05
        beta_velocity = 0.9
        beta_rmsprop = 0.9

        velocity_theta = 0.0
        s_theta = 0.0
        rms_offset = 1e-6  # small offset to make sure we don't take 1/sqrt(very small) in weight update

        # Initialize params for tracking best solution
        best_primary = np.inf  # minimizing f so want it to be lowest possible
        best_theta = np.copy(theta_init)

        # Set up the gradient function using autograd.
        df_dtheta_fn = grad(self.wrapped_primary_objective, argnum=0)

        # Start gradient descent
        if verbose:
            print(f"Have {n_iters_tot} iterations")

        for iter_index in range(n_iters_tot):
            if verbose:
                if iter_index % 10 == 0:
                    print(f"Iter: {iter_index}")
            primary_val = self.wrapped_primary_objective(theta)

            if debug:
                print("iter,f,theta:", iter_index, primary_val, theta)
                print()

            # Check if this is best value so far
            if primary_val < best_primary:
                best_primary = primary_val
                best_theta = np.copy(theta)

            # if nans or infs appear in any quantities,
            # then stop gradient descent and return NSF
            if np.isinf(primary_val) or np.isinf(theta).any():
                warning_msg = (
                    "Warning: a nan or inf was found during "
                    "gradient descent. Stopping prematurely "
                    "and returning best solution so far"
                )
                warnings.warn(warning_msg)
                break

            # Calculate df/dtheta
            df_theta = df_dtheta_fn(theta)

            # Momementum term
            velocity_theta = (
                beta_velocity * velocity_theta + (1.0 - beta_velocity) * df_theta
            )

            # RMS prop term
            s_theta = beta_rmsprop * s_theta + (1.0 - beta_rmsprop) * pow(df_theta, 2)

            # bias-correction
            velocity_theta /= 1 - pow(beta_velocity, iter_index + 1)
            s_theta /= 1 - pow(beta_rmsprop, iter_index + 1)

            # update weights
            theta -= (
                alpha_theta * velocity_theta / (np.sqrt(s_theta) + rms_offset)
            )  # gradient descent
        return best_theta
