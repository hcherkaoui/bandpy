""" Define all the MAB bandit environments availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import collections
import numpy as np

from ._base import BanditEnvBase


class BernoulliKBandit(BanditEnvBase):
    """BernoulliKBandit class to define a Bernoulli bandit with K arms.

    Parameters
    ----------
    p : array-like of float, each float is the Bernoulli probability
        corresponding to the k-th arm.
    T : int, the iteration finite horizon.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, p, T, seed=None):
        """Init."""

        super().__init__(T=T, seed=seed)

        msg = ("BernoulliKBandit should be instanciated with a 1 dim "
               "array-like of K probabilities.")

        if isinstance(p, collections.abc.Sequence):
            self.p = np.array(p, dtype=float)

            if self.p.ndim != 1:
                raise ValueError(msg)

            self.K = len(p)

        else:
            raise ValueError(msg)

        self.best_arm = np.argmax(self.p)
        self.best_reward = np.max(self.p)

    def update_agent_stats(self, name_agent, y, no_noise_y):
        y_max = np.max(self.mu)
        y_min = np.min(self.mu)
        self._update_agent_stats(name_agent, y, no_noise_y, y_max, y_min)

    def compute_reward(self, name_agent, k):
        y = no_noise_y = int(self.rng.rand() <= self.p[k])
        return y, no_noise_y


class GaussianKBandit(BanditEnvBase):
    """'GaussianKBandit' class to define a Gaussian bandit with K arms.

    Parameters
    ----------
    mu : array-like of float, each float is the mean corresponding to the
        k-th arm.
    sigma : array-like of float, each float is the standard-deviation
        corresponding to the k-th arm.
    T : int, the iteration finite horizon.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, mu, sigma, T, seed=None):
        """Init."""

        super().__init__(T=T, seed=seed)

        msg = ("'GaussianKBandit' should be instanciated with a 1 dim "
               "array-like of K means.")

        if isinstance(mu, collections.abc.Sequence):
            self.mu = np.array(mu, dtype=float)

            if self.mu.ndim != 1:
                raise ValueError(msg)

            self.K = len(self.mu)

        else:
            raise ValueError(msg)

        msg = ("'GaussianKBandit' should be instanciated with a 1 dim "
               "array-like of K standard-deviations.")

        if isinstance(sigma, collections.abc.Sequence):
            self.sigma = np.array(sigma, dtype=float)

            if self.sigma.ndim != 1:
                raise ValueError(msg)

        if self.mu.shape != self.sigma.shape:
            raise ValueError(f"'mu' and 'sigma' should have the same "
                             f"dimension, got {self.mu.shape}, resp. "
                             f"{self.sigma.shape}")

        self.best_arm = np.argmax(self.mu)
        self.best_reward = np.max(self.mu)

    def update_agent_stats(self, name_agent, y, no_noise_y):
        y_max = np.max(self.mu)
        y_min = np.min(self.mu)
        self._update_agent_stats(name_agent, y, no_noise_y, y_max, y_min)

    def compute_reward(self, name_agent, k):
        y = self.mu[k] + self.sigma[k] * self.rng.randn()
        no_noise_y = self.mu[k]
        return y, no_noise_y
