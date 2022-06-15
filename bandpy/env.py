""" Define all the bandit environments available in bandpy.
"""
import collections
import numpy as np
from .utils import check_random_state


class BernoulliKBandit:
    """BernoulliKBandit class to define a Bernoulli bandit with K arms.

    Parameters
    ----------
    p : array-like of float, each float is the Bernoulli probability
        corresponding to the k-th arm.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, p, horizon, seed=None):
        """Init."""

        msg = ("BernoulliKBandit should be instanciated with a 1 dim "
               "array-like of K probabilities.")

        if isinstance(p, collections.Sequence):
            self.p = np.array(p, dtype=float)

            if self.p.ndim != 1:
                raise ValueError(msg)

            self.K = len(p)

        else:
            raise ValueError(msg)

        self.horizon = horizon

        self.random_state = check_random_state(seed)

        self.best_arm = np.argmax(self.p)
        self.best_reward = np.max(self.p)

        self.n = 0
        self.total_reward = 0.0

    def step(self, k):
        """Pull the k-th arm."""
        reward = self.random_state.binomial(1, self.p[k])
        self.n += 1
        self.total_reward += reward

        observation = {'arm_pulled': k, 'reward': reward}
        done = True
        info = {'n_arms': self.K}

        done = False
        if self.horizon <= self.n:
            done = True

        return observation, reward, done, info

    def regret(self):
        """Regret function."""
        return self.n * np.max(self.p) - self.total_reward
