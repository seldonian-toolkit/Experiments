import autograd.numpy as np
from .baselines import RLExperimentBaseline


class BaseFittedQBaseline(RLExperimentBaseline):
    def __init__(
        self,
        model_name,
        regressor_class,
        policy,
        num_iters=100,
        env_kwargs={"gamma": 1.0},
        regressor_kwargs={},
    ):
        """Base class for fitted-Q RL baselines. All methods that raise 
        NotImplementedError must be implemented in child classes. Any method
        here can be overridden in a child class. 

        :param regressor_class: The class (not object) for the regressor.
            Must have a fit() method which takes features and lables as first two args.
        :param policy: A seldonian.RL.Agents.Policies.Policy object (or child of).
        :param num_iters: The number of iterations to run fitted Q.
        :param env_kwargs: A dictionary containing environment-specific key,value pairs.
            Must contain a "gamma" key with a float value between 0 and 1. This is used
            for computing the Q target.
        """
        super().__init__(model_name, policy, env_kwargs)
        self.regressor_class = regressor_class
        self.regressor_kwargs = regressor_kwargs
        self.num_iters = num_iters
        self.last_greedy_actions = []

    def get_probs_from_observations_and_actions(
        self, theta, observations, actions, behavior_action_probs
    ):
        """
        A wrapper for obtaining the action probabilities for each timestep a single episode.
        These are needed for getting the IS estimates for each new policy proposed in the experiment trials.

        :param theta: Weights of the new policy
        :param observations: An array of the observations at each timestep in the episode
        :param actions: An array of the actions at each timestep in the episode
        :param behavior_action_probs: An array of the action probabilities of the behavior policy
            at each timestep in the episode

        :return: Action probabilities under the new policy (parameterized by theta)
        """
        self.set_new_params(theta)
        return self.policy.get_probs_from_observations_and_actions(
            observations, actions, behavior_action_probs
        )

    def set_new_params(self, weights):
        return self.policy.set_new_params(weights)

    def get_policy_params(self):
        return self.policy.get_params()

    def train(self, dataset, **kwargs):
        """ """
        # reset policy to all zero weights
        self.reset_policy_params()
        transitions = self.get_transitions(dataset.episodes)  # (o,a,o',r) tuples
        for i in range(self.num_iters):
            if i == 0:
                X, y = self.make_regression_dataset(transitions, make_X=True)
            else:
                _, y = self.make_regression_dataset(transitions, make_X=False)
            # Instantiate a new regressor in each iteration so it has no memory of the previous fit
            self.regressor = self.instantiate_regressor()
            self.regressor.fit(X, y)
            # Update Q function using results of the regression
            self.update_Q_weights()
            if self.stopping_criteria_met():
                break
        return self.get_policy_params()

    def reset_policy_params():
        raise NotImplementedError("Implement this method in a child class")

    def get_transitions(self, episodes):
        transitions = []
        for ep in episodes:
            for ii in range(len(ep.observations)):
                next_obs = self.get_next_obs(ep.observations, ii)
                tup = (ep.observations[ii], ep.actions[ii], next_obs, ep.rewards[ii])
                transitions.append(tup)
        return transitions

    def get_next_obs(self, observations, index):
        """Get the next observation, o', from a given transition. Sometimes this
        is trivial, but often not if there is a finite time horizon, for example.

        :param observations: Array of observations for a given episode
        :param index: The index of the current observation:

        :return: next observation.
        """
        raise NotImplementedError("Implement this method in a child class")

    def make_regression_dataset(self, transitions, make_X=False):
        """Make the features and labels for the regression algorithm to fit.
        We don't need to remake X (s,a) every time because it never changes. y does
        change upon each step, so we need to make a new y for each iteration of fitted Q.

        :param transitions: List of transition tuples (s,a,s',r) for the whole dataset
        :param make_X: A boolean flag indicating whether we need to make the features, X
            from the transition tuples.

        :return: X,y - X is a 2D numpy ndarray and y is a 1D numpy ndarray.
        """
        observations, actions, next_observations, rewards = zip(*transitions)
        if make_X:
            X = self.make_X(observations, actions)
        else:
            X = None
        y = self.make_y(rewards, next_observations)
        return X, y

    def make_X(self, observations, actions):
        """Make the feature array that will be used to train the regressor."""
        raise NotImplementedError("Implement this method in a child class")

    def make_y(self, rewards, next_observations):
        """Make the label array that will be used to train the regressor.
        One could speed this up if their get_target() method can be
        vectorized. That depends on if their get_max_q() method can be
        vectorized.
        """
        y = np.zeros(len(rewards))
        for ii in range(len(rewards)):
            y[ii] = self.get_target(rewards[ii], next_observations[ii])
        return y

    def get_target(self, reward, next_obs):
        """Get the Q target, which is the label for training the regressor

        :param reward: Scalar reward
        :param next_obs: The state that was transitioned to.

        :return: Q target - a real number.
        """
        return reward + self.gamma * self.get_max_q(next_obs)

    def get_max_q(self, obs):
        """Get the max of the q function
        over all possible actions in a given observation.

        For evaluating the max_a' { Q(s_t+1,a') }
        term in the Q target.
        """
        raise NotImplementedError("Implement this method in a child class")

    def instantiate_regressor(self):
        """Create the regressor object and return it.
        This should be an instance of the self.regressor_class class, instantiated with
        whatever parameters you need.

        :return: Regressor object, ready to be trained.
        """
        raise NotImplementedError("Implement this method in a child class")

    def update_Q_weights(self):
        """Update Q function weights given results of the regressor."""
        raise NotImplementedError("Implement this method in a child class")

    def stopping_criteria_met(self):
        """Define stopping criteria. Return True if stopping criteria are met.
        If you want to run for self.num_iters and don't want any stopping criteria
        just return False in your child-class implementation of this method.
        """
        raise NotImplementedError("Implement this method in a child class")



class ExactTabularFittedQBaseline(BaseFittedQBaseline):
    def __init__(
        self, model_name, regressor_class, policy, num_iters=100, env_kwargs={'gamma':1.0}, regressor_kwargs={}
    ):
        """Implements fitted-Q RL baseline where the policy is a Q table.
        Uses the regressor weights to update the Q table. Works for parametric
        models. The features of the regression problem are the one-hot vectors
        of the (observation,action) pairs. 

        Env kwargs needs to include the following key,value pairs:
            "gamma": float
            "num_observations": int
            "num_actions": int
            "terminal_obs": int (the terminal state)

        """
        super().__init__(
            model_name=model_name,
            regressor_class=regressor_class,
            policy=policy,
            num_iters=num_iters,
            env_kwargs=env_kwargs,
            regressor_kwargs=regressor_kwargs,
        )
        self.num_observations = env_kwargs["num_observations"]
        self.num_actions = env_kwargs["num_actions"]
        self.terminal_obs = env_kwargs["terminal_observation"]

    def reset_policy_params(self):
        self.policy.set_new_params(np.zeros((self.num_observations, self.num_actions)))

    def get_next_obs(self, observations, index):
        """Get the next observation, o', from a given transition. Sometimes this
        is trivial, but often not if there is a finite time horizon, for example.

        :param observations: Array of observations for a given episode
        :param index: The index of the current observation:

        :return: next observation.
        """
        if index == len(observations) - 1:
            next_obs = self.terminal_obs
        else:
            next_obs = observations[index + 1]
        return next_obs

    def make_X(self, observations, actions):
        """Make the feature array that will be used to train the regressor."""
        X = np.array(list(map(self.one_hot_encode, observations, actions)))
        return X

    def one_hot_encode(self, o, a):
        """Turn an observation,action pair into a one-hot vector (1D numpy.ndarray)"""
        vector = np.zeros(self.num_observations * self.num_actions)
        vector[o * self.num_actions + a] = 1
        return vector

    def get_max_q(self, obs):
        """Get the max q function value given
        an observation over all actions in that observation.

        For evaluating the max_a' Q(s_t+1,a')
        term in the target.
        """
        return max(self.policy.get_params()[obs])

    def instantiate_regressor(self):
        """Create the regressor object and return it.
        This should be an instance of the self.regressor_class class, instantiated with
        whatever parameters you need.

        :return: Regressor object, ready to be trained.
        """
        regressor = self.regressor_class(fit_intercept=False)
        return regressor

    def update_Q_weights(self):
        """Update Q function weights given results of the regressor."""
        fitted_weights = self.get_regressor_weights()
        self.set_q_table(fitted_weights)

    def get_regressor_weights(self):
        """Get out the weights
        from the regressor, reshaping so they
        have same shape as Q table.
        """
        native_weights = self.regressor.coef_
        return native_weights.reshape(self.num_observations, self.num_actions)

    def set_q_table(self, weights):
        """Set the Q table parameters"""
        self.set_new_params(weights)

    def stopping_criteria_met(self):
        """If the greedy actions in each observation
        are not changing from iteration to iteration
        then we can stop the algorithm and return the optimal solution.
        Should keep track of last few greedy actions. 
        """
        current_greedy_actions = np.array([np.argmax(self.policy.get_params()[obs]) for obs in range(self.num_observations)])
        if len(self.last_greedy_actions) == 0:
            self.last_greedy_actions = current_greedy_actions
            return False
        if all([current_greedy_actions[ii] == self.last_greedy_actions[ii] for ii in range(len(current_greedy_actions))]):
            self.last_greedy_actions = current_greedy_actions
            return True

        self.last_greedy_actions = current_greedy_actions
        return False

class ApproximateTabularFittedQBaseline(ExactTabularFittedQBaseline):
    def __init__(self,model_name,regressor_class,policy,num_iters=100,env_kwargs={'gamma':1.0},regressor_kwargs={}):
        """Implements fitted-Q RL baseline for a Q table, 
        but uses the fitted regressor to approximate the Q values.
        Useful for nonparametric regressors. The features of the 
        regression problem are the one-hot vectors
        of the (observation,action) pairs. 
        """
        super().__init__(
            model_name=model_name,
            regressor_class=regressor_class,
            policy=policy,
            num_iters=num_iters,
            env_kwargs=env_kwargs,
            regressor_kwargs=regressor_kwargs
        )

    def instantiate_regressor(self):
        """Create the regressor object and return it.
        This should be an instance of the self.regressor_class class, instantiated with
        whatever parameters you need.

        :return: Regressor object, ready to be trained.
        """
        regressor = self.regressor_class()
        return regressor
    
    def get_regressor_weights(self):
        """ Approximates Q table by passing each possible 
        one-hot encoding of (observation,action) pairs
        through the regressor's forward pass. """
        vecs = []
        for i in range(self.num_observations*self.num_actions):
            vec = np.zeros(self.num_observations*self.num_actions)
            vec[i] = 1
            vecs.append(vec)
        Q = self.regressor.predict(vecs)
        return Q.reshape(self.num_observations,self.num_actions)
