""" Define all the bandit environments availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import collections
import numpy as np
from .utils import check_random_state


class BanditEnv():
    """ Virtual class of a Bandit environment. """

    def step(self, k):
        raise NotImplementedError


class BernoulliKBandit(BanditEnv):
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

        msg = ("BernoulliKBandit should be instanciated with a 1 dim "
               "array-like of K probabilities.")

        if isinstance(p, collections.abc.Sequence):
            self.p = np.array(p, dtype=float)

            if self.p.ndim != 1:
                raise ValueError(msg)

            self.K = len(p)

        else:
            raise ValueError(msg)

        self.T = T

        self.rng = check_random_state(seed)

        self.best_arm = np.argmax(self.p)
        self.best_reward = np.max(self.p)

        self.t = 0
        self.total_reward = 0.0

    def step(self, k):
        """Pull the k-th arm."""
        self.t += 1
        reward = int(self.rng.rand() <= self.p[k])
        self.total_reward += reward

        done = False
        if self.T <= self.t:
            done = True

        to_return = [{'arm_pulled': k, 'reward': reward, 't': self.t},
                     reward,
                     done,
                     {'n_arms': self.K},
                     ]

        return to_return

    def regret(self):
        """Expected regret."""
        return np.max(self.p) - self.total_reward / self.t


class GaussianKBandit(BanditEnv):
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

        self.T = T

        self.rng = check_random_state(seed)

        self.best_arm = np.argmax(self.mu)
        self.best_reward = np.max(self.mu)

        self.t = 0
        self.total_reward = 0.0

    def step(self, k):
        """Pull the k-th arm."""
        self.t += 1
        reward = self.mu[k] + self.sigma[k] * self.rng.randn()
        self.total_reward += reward

        done = False
        if self.T <= self.t:
            done = True

        to_return = [{'arm_pulled': k, 'reward': reward, 't': self.t},
                     reward,
                     done,
                     {'n_arms': self.K},
                     ]

        return to_return

    def regret(self):
        """Expected regret."""
        return np.max(self.mu) - self.total_reward / self.t


class LinearBandit2D(BanditEnv):
    """'LinearBandit2D' class to define a Linear Bandit with 3 arms ([1, 0],
    [0, 1], [cos(delta), sin(delta)]) with delta in ]0.0, pi[ and
    theta = [2, 0]. The reward is defined as 'r = theta.T.dot(x_k) + eta' with
    eta drawn from a centered normalized Gaussian noise.

    Parameters
    ----------
    T : int, the iteration finite horizon.
    delta : float, default=0.01, angle for the third arm (supposed to be closed
        to the optimal arm x_2)
    """

    def __init__(self, T, delta=0.01):
        """Init."""
        if not ((delta > 0.0) and (delta < np.pi)):
            raise ValueError(f"delta should belongs to ]0.0, pi[, "
                             f"got {delta}")

        # 3 arms defined by x_k = [x, y]
        x_0 = np.array([1.0, 0.0])  # best arm
        x_1 = np.array([0.0, 1.0])
        x_2 = np.array([np.cos(delta), np.sin(delta)])
        self.theta = np.array([2.0, 0.0])
        self.arms = [x_0, x_1, x_2]

        self.T = T

        self.best_arm = 0
        self.best_reward = self.theta.dot(self.arms[self.best_arm])

        self.t = 0
        self.total_reward = 0.0

    def step(self, k):
        """Pull the k-th arm."""
        self.t += 1
        reward = self.theta.dot(self.arms[k]) + self.rng.randn()
        self.total_reward += reward

        done = False
        if self.T <= self.t:
            done = True

        to_return = [{'arm_pulled': k, 'reward': reward, 't': self.t},
                     reward,
                     done,
                     {'n_arms': self.K},
                     ]

        return to_return

    def regret(self):
        """Expected regret."""
        return np.max(self.mu) - self.total_reward / self.t
