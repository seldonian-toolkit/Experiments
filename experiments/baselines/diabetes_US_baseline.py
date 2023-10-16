import autograd.numpy as np
import cma
import cma
from seldonian.utils.io_utils import cmaes_logger
from seldonian.RL.Agents.Policies.SimglucosePolicyFixedArea import (
    SigmoidPolicyFixedArea,
)


class RLDiabetesUSAgentBaseline:
    def __init__(
        self,
        initial_solution,
        env_kwargs,
        bb_crmin=5.0,
        bb_crmax=15.0,
        bb_cfmin=15.0,
        bb_cfmax=25.0,
        cr_shrink_factor=np.sqrt(3),
        cf_shrink_factor=np.sqrt(3),
    ):
        """Implements an RL baseline that uses importance sampling
        with unequal support (US) with a fixed area policy"""
        super().__init__()
        self.model_name = "diabetes_us"
        self.initial_solution = initial_solution
        self.env_kwargs = env_kwargs
        self.policy = SigmoidPolicyFixedArea(
            bb_crmin=bb_crmin,
            bb_crmax=bb_crmax,
            bb_cfmin=bb_cfmin,
            bb_cfmax=bb_cfmax,
            cr_shrink_factor=cr_shrink_factor,
            cf_shrink_factor=cf_shrink_factor,
        )

    def set_new_params(self, new_params):
        """Set the parameters of the agent

        :param new_params: array of weights
        """
        self.policy.set_new_params(new_params)

    def train(self, dataset, **kwargs):
        """
        Run CMA-ES starting with a random initial policy parameterization
        """
        self.episodes = dataset.episodes
        n_eps = len(self.episodes)
        theta_init = self.initial_solution
        crmin_init, crmax_init, cfmin_init, cfmax_init = self.policy.theta2crcf(
            theta_init
        )
        opts = {}
        if "seed" in kwargs:
            opts["seed"] = kwargs["seed"]

        if "sigma0" in kwargs:
            sigma0 = kwargs["sigma0"]
        else:
            sigma0 = 5

        es = cma.CMAEvolutionStrategy(theta_init, sigma0, opts)
        # minimize the primary objective function,
        # which is the expected return. So we want to minimize
        # the negative expected return
        es.optimize(self.primary_objective_fn, callback=None)
        solution = es.result.xbest
        crmin_sol, crmax_sol, cfmin_sol, cfmax_sol = self.policy.theta2crcf(solution)
        if (solution is None) or (not all(np.isfinite(solution))):
            solution = "NSF"
        return solution

    def primary_objective_fn(self, theta):
        """This is the function we want to minimize.
        In RL, we want to maximize the expected return
        so we need to minimize the negative expected return.
        """
        crmin, crmax, cfmin, cfmax = self.policy.theta2crcf(theta)
        returns_inside_theta_box = []
        for ii, ep in enumerate(self.episodes):
            cr_b, cf_b = ep.actions[0]  # behavior policy action
            primary_return = ep.rewards[0]  # one reward per episode, so reward=return
            if (crmin <= cr_b <= crmax) and (cfmin <= cf_b <= cfmax):
                returns_inside_theta_box.append(primary_return)
        f = np.mean(returns_inside_theta_box)
        return -1.0 * f
