""" Define all the bandit environments availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import collections
import numpy as np
from .base import BanditEnv
from .utils import check_random_state


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

    def compute_reward(self, name_agent, k):
        return int(self.rng.rand() <= self.p[k])


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

    def compute_reward(self, name_agent, k):
        return self.mu[k] + self.sigma[k] * self.rng.randn()


class LinearBandit(BanditEnv):
    """Abstract calss for 'LinearBandit' class to define a Linear Bandit in
    dimension 'd'The reward is defined as 'r = theta.T.dot(x_k) + noise' with
    noise drawn from a centered Gaussian distribution.

    Parameters
    ----------
    T : int, the iteration finite horizon.
    arms : list of np.array, list of arms.
    theta : np.array, theta parameter.
    seed : np.random.RandomState instance, the random seed.
    """
    def __init__(self, T, arms, theta, sigma=1.0, seed=None):
        """Init."""

        super().__init__(T=T, seed=seed)

        self.arms = arms
        self.K = len(self.arms)

        self.theta = theta
        self.sigma = sigma

        all_rewards = [float(self.theta.T.dot(x_k)) for x_k in self.arms]
        self.best_arm = np.argmax(all_rewards)
        self.best_reward = np.max(all_rewards)

    def compute_reward(self, name_agent, k):
        x_k = self.arms[k].reshape((self.d, 1))
        r = float(x_k.T.dot(self.theta))
        noise = float(self.sigma * self.rng.randn())
        return r + noise


class CanonicalLinearBandit(LinearBandit):
    """'LinearBandit' class to define a Linear Bandit in dimension 'd' with d+1
    arms ([1, 0, 0, ...], [0, 1, 0, ...], [cos(delta), sin(delta), ...]) with
    delta in ]0.0, pi[ and theta = [2, 0, ...]. The reward is defined as
    'r = theta.T.dot(x_k) + noise' with noise drawn from a centered Gaussian
    distribution.

    Parameters
    ----------
    T : int, the iteration finite horizon.
    d : int, dimension of the problem.
    delta : float, default=0.01, angle for the third arm (supposed to be closed
        to the optimal arm x_2)
    """
    def __init__(self, T, d, delta, sigma=1.0, seed=None):
        """Init."""

        if d < 2:
            raise ValueError(f"Dimendion 'd' should be >=2, got {d}")

        if not ((delta > 0.0) and (delta < np.pi)):
            raise ValueError(f"delta should belongs to ]0.0, pi[, "
                             f"got {delta}")

        self.d = d

        arms = []
        for i in range(self.d):
            x_ = np.zeros((self.d, 1))
            x_[i] = 1.0
            arms.append(x_)

        nearly_opt_x_k = np.zeros((self.d, 1))
        nearly_opt_x_k[0] = np.cos(delta)
        nearly_opt_x_k[1] = np.sin(delta)
        arms.append(np.array(nearly_opt_x_k))

        theta = np.zeros((self.d, 1))
        theta[0] = 2.0

        super().__init__(T=T, arms=arms, theta=theta, sigma=sigma, seed=seed)


class RandomLinearBandit(LinearBandit):
    """'RandomLinearBandit' class to define a Linear Bandit in dimension 'd'
    with K random Gaussian arms and a Gaussian random theta. The reward is
    defined as 'r = theta.T.dot(x_k) + noise' with noise drawn from a centered
    Gaussian distribution.

    Parameters
    ----------
    T : int, the iteration finite horizon.
    d : int, dimension of the problem.
    K : int, number of arms.
    """
    def __init__(self, T, d, K, sigma=1.0, seed=None):
        """Init."""

        if d < 2:
            raise ValueError(f"Dimendion 'd' should be >=2, got {d}")

        self.d = d

        rng = check_random_state(seed)

        arms = []
        for _ in range(K):
            arms.append(rng.randn(self.d))

        theta = rng.randn(self.d)

        super().__init__(T=T, arms=arms, theta=theta, sigma=sigma, seed=rng)
