""" Define all the bandit environments availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import collections
import numpy as np
from scipy import optimize

from .base import BanditEnv
from .loaders import movie_lens_loader, yahoo_loader
from .checks import (check_random_state, check_K_arms_arm_entries,
                     check_thetas_and_n_thetas)
from .criterions import f_neg_scalar_prod, grad_neg_scalar_prod


DEFAULT_DIR_DATASET_MOVIELENS = ("/mnt/c/Users/hwx1143141/Desktop/datasets/"
                                 "ml-latest-small")
DEFAULT_DIR_DATASET_YAHOO = ("/mnt/c/Users/hwx1143141/Desktop/datasets/yahoo/"
                             "ltrc_yahoo/")

NB_MAX_USERS_MOVIELENS = 610
NB_MAX_USERS_YAHOO = 1266


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
    sigma : float, standard deviation of the noise.
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
    sigma : float, standard deviation of the noise.
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


class ClusteredGaussianLinearBandit(BanditEnv):
    """'ClusteredGaussianLinearBandit' class to define a Linear Bandit in
    dimension 'd'. The arms and the theta are drawn following a Gaussian. The
    reward is defined as 'r = theta_l_k.T.dot(x_k) + noise' with noise drawn
    from a centered Gaussian distribution.

    Parameters
    ----------
    N :
    T : int, the iteration finite horizon.
    d : int, dimension of the problem.
    K :
    arms :
    arm_entries :
    n_thetas : int, number of theta for the model.
    thetas :
    sigma : float, standard deviation of the noise.
    theta_offset :
    shuffle_labels :
    seed :
    """

    def __init__(self, N, T, d, K=None, arms=None, arm_entries=None,
                 n_thetas=None, thetas=None, sigma=1.0, theta_offset=0.0,
                 shuffle_labels=True, seed=None):
        """Init."""
        if d < 2:
            raise ValueError(f"Dimendion 'd' should be >= 2, got {d}")

        thetas, n_thetas = check_thetas_and_n_thetas(d=d, thetas=thetas,
                                                     n_thetas=n_thetas,
                                                     theta_offset=theta_offset,
                                                     seed=seed)
        self.thetas = thetas
        self.n_thetas = n_thetas

        parameters = check_K_arms_arm_entries(d=d,
                                              arms=arms,
                                              arm_entries=arm_entries,
                                              K=K,
                                              seed=seed)
        return_arm_index, arms, arm_entries, K = parameters

        self.return_arm_index = return_arm_index
        self.arms = arms
        self.arm_entries = arm_entries
        self.K = K

        self.N = N
        self.d = d

        self.shuffle_labels = shuffle_labels
        self.rng = check_random_state(seed)

        self.theta_per_agent = self._assign_agent_models()

        self.sigma = sigma

        # deactivate best_arm / best_reward
        self.best_arm = dict()
        self.best_reward = dict()
        self.worst_reward = dict()
        for i, theta in enumerate(self.thetas):

            if self.return_arm_index:

                all_rewards = [theta.T.dot(arm) for arm in self.arms]

                self.best_reward[i] = np.max(all_rewards)
                self.worst_reward[i] = np.min(all_rewards)
                self.best_arm[i] = np.argmax(all_rewards)

            else:

                def f(x):
                    return f_neg_scalar_prod(x, theta)

                def grad(x):
                    return grad_neg_scalar_prod(x, theta)

                x0 = np.zeros_like(theta)
                bounds = [(np.min(entry_vals), np.max(entry_vals))
                          for entry_vals in self.arm_entries.values()]

                res = optimize.minimize(fun=f, jac=grad, x0=x0,
                                        method='L-BFGS-B', bounds=bounds)
                best_reward_ = - res.fun
                best_arm_ = res.x

                def f(x):
                    return - f_neg_scalar_prod(x, theta)

                def grad(x):
                    return - grad_neg_scalar_prod(x, theta)

                res = optimize.minimize(fun=f, jac=grad, x0=x0,
                                        method='L-BFGS-B', bounds=bounds)
                worst_reward_ = res.fun

                self.best_reward[i] = best_reward_
                self.worst_reward[i] = worst_reward_
                self.best_arm[i] = best_arm_

        super().__init__(T=T, seed=self.rng)

    def _assign_agent_models(self):
        """Assign a theta for each agent."""

        theta_idx = []  # [0 0 0 ... 1 1 ... 2 ... 2 2]
        for theta_i in range(self.n_thetas):
            theta_idx += [theta_i] * int(self.N / self.n_thetas)
        theta_idx += [theta_i] * (self.N - len(theta_idx))

        if self.shuffle_labels:
            self.rng.shuffle(theta_idx)

        theta_per_agent = dict()
        for i, theta_i in enumerate(theta_idx):
            theta_per_agent[f"agent_{i}"] = theta_i

        return theta_per_agent

    def compute_reward(self, agent_name, k_or_arm):
        """Compute the reward associated to the given arm or arm-index."""

        if isinstance(k_or_arm, (int, np.integer)):
            x_k = self.arms[k_or_arm].reshape((self.d, 1))

        else:
            x_k = k_or_arm.reshape((self.d, 1))

        theta = self.thetas[self.theta_per_agent[agent_name]]

        r = float(x_k.T.dot(theta))
        noise = float(self.sigma * self.rng.randn())

        return r + noise


class DatasetEnv(BanditEnv):
    """ Environment based on a real dataset. """

    def __init__(self, T, N, K, d, dirname, sigma=1.0, seed=None):
        """Init."""
        self.dirname = dirname

        self.d = d
        self.K = K
        self.N = N

        self.T = T
        self.t = 1

        self.S_t = dict()
        self.best_S_t = dict()
        self.worst_S_t = dict()

        self.seed = seed
        self.rng = check_random_state(seed)

        internals = self.load_dataset(self.dirname, self.N, self.K, self.d)
        self.data, self.arms, self.agent_i_to_env_agent_i = internals

        self.sigma = sigma

        self.best_arm = dict()
        self.best_reward, self.worst_reward = dict(), dict()
        for i in range(self.N):

            id = self.agent_i_to_env_agent_i[f"agent_{i}"]
            all_rewards = [self.get_r(id, k) for k in range(self.K)]

            self.best_reward[f"agent_{i}"] = np.max(all_rewards)
            self.best_arm[f"agent_{i}"] = np.argmax(all_rewards)
            self.worst_reward[f"agent_{i}"] = np.min(all_rewards)

    def get_r(self, agent_id, arm_id):
        """Get the reward for agent `agent_id` and arm `arm_id`."""
        filter_entry = ((self.data.agent_id == agent_id)
                        & (self.data.arm_id == arm_id))
        if not self.data[filter_entry].empty:
            return float(self.data[filter_entry].reward)
        else:
            raise ValueError(f"DataFrame does not have reward for 'agent' = "
                             f"{agent_id} and 'arm' = {arm_id}.")

    def reset(self, seed=np.NaN):
        """Reset the environment (and the randomness if seed is not NaN)."""
        self.t = 1
        self.S_t = dict()
        self.best_S_t = dict()
        self.worst_S_t = dict()
        if not np.isnan(seed):
            self.rng = check_random_state(seed)

    def update_agent_total_rewards(self, name_agent, r):
        """Update S_t = sum_{s=1}^t r_s."""
        if name_agent in self.S_t:
            self.S_t[name_agent] += r
        else:
            self.S_t[name_agent] = r

        if name_agent in self.best_S_t:
            self.best_S_t[name_agent] += self.best_reward[name_agent]
        else:
            self.best_S_t[name_agent] = self.best_reward[name_agent]

        if name_agent in self.worst_S_t:
            self.worst_S_t[name_agent] += self.worst_reward[name_agent]
        else:
            self.worst_S_t[name_agent] = self.worst_reward[name_agent]

    def regret(self):
        """Expected regret."""
        if not self.S_t:
            return np.NaN

        else:
            name_agents = self.S_t.keys()
            S_t = np.array(list(self.S_t.values()), dtype=float)
            best_S_t = np.array(list(self.best_S_t.values()), dtype=float)
            regrets = (best_S_t - S_t) / self.t
            return dict(zip(name_agents, regrets))

    def compute_reward(self, agent_name, k):
        id = self.agent_i_to_env_agent_i[agent_name]
        return self.get_r(id, k) + float(self.sigma * self.rng.randn())


class MovieLensEnv(DatasetEnv):
    """ Movie-Lens Bandit environment. """

    def __init__(self, T, N, K, d, sigma=1.0,
                 dirname=DEFAULT_DIR_DATASET_MOVIELENS, seed=None):
        """Init."""
        if N > NB_MAX_USERS_MOVIELENS:
            raise ValueError(f"Maximum agent number is "
                             f"{NB_MAX_USERS_MOVIELENS}, got N={N}")

        super().__init__(T=T, N=N, K=K, d=d, sigma=sigma, dirname=dirname,
                         seed=seed)

    def load_dataset(self, dirname, N, K, d):
        return movie_lens_loader(dirname, N, K, d)


class YahooEnv(DatasetEnv):
    """ Yahoo Bandit environment. """

    def __init__(self, T, N, K, d, sigma=1.0,
                 dirname=DEFAULT_DIR_DATASET_YAHOO, n_clusters_k_means=100,
                 seed=None):
        """Init."""
        if N > NB_MAX_USERS_YAHOO:
            raise ValueError(f"Maximum agent number is "
                             f"{NB_MAX_USERS_YAHOO}, got N={N}")

        self.n_clusters_k_means = n_clusters_k_means

        super().__init__(T=T, N=N, K=K, d=d, sigma=sigma, dirname=dirname,
                         seed=seed)

    def load_dataset(self, dirname, N, K, d):
        return yahoo_loader(dirname, N, K, d,
                            n_clusters_k_means=self.n_clusters_k_means,
                            seed=self.seed)
