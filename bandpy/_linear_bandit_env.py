""" Define all the linear bandit environments availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from scipy import optimize

from ._base import BanditEnvBase
from ._loaders import movie_lens_loader, yahoo_loader
from ._checks import (check_random_state, check_K_arms_arm_entries,
                      check_thetas_and_n_thetas)
from ._criterions import f_neg_scalar_prod, grad_neg_scalar_prod


DEFAULT_DIR_DATASET_MOVIELENS = ("/mnt/c/Users/hwx1143141/Desktop/datasets/"
                                 "ml-latest-small")
DEFAULT_DIR_DATASET_YAHOO = ("/mnt/c/Users/hwx1143141/Desktop/datasets/yahoo/"
                             "ltrc_yahoo/")

NB_MAX_USERS_MOVIELENS = 610
NB_MAX_USERS_YAHOO = 1266


class ClusteredGaussianLinearBandit(BanditEnvBase):
    """'ClusteredGaussianLinearBandit' class to define a Linear Bandit in
    dimension 'd'. The arms and the theta are drawn following a Gaussian. The
    reward is defined as 'r = theta_l_k.T.dot(x_k) + noise' with noise drawn
    from a centered Gaussian distribution.

    Parameters
    ----------
    N :
    T : int, the iteration finite horizon.
    d : int, dimension of the problem.
    K : int,
    arms : list of array,
    arm_entries : dict of list,
    n_thetas : int, number of theta for the model.
    thetas : list of array,
    sigma : float, standard deviation of the noise.
    theta_offset : float,
    shuffle_labels : bool,
    seed : np.random.RandomState instance, the random seed.
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

    def update_agent_stats(self, name_agent, y, no_noise_y):
        """Update all statistic as listed in __init__ doc.

        Parameters
        ----------
        """

        theta_idx = self.theta_per_agent[name_agent]

        y_max = self.best_reward[theta_idx]
        y_min = self.worst_reward[theta_idx]

        self._update_agent_stats(name_agent, y, no_noise_y, y_max, y_min)

    def compute_reward(self, agent_name, k_or_arm):
        """Compute the reward associated to the given arm or arm-index."""

        if isinstance(k_or_arm, (int, np.integer)):
            x_k = self.arms[k_or_arm].reshape((self.d, 1))

        else:
            x_k = k_or_arm.reshape((self.d, 1))

        theta = self.thetas[self.theta_per_agent[agent_name]]

        no_noise_y = float(x_k.T.dot(theta))
        noise = float(self.sigma * self.rng.randn())

        return noise + no_noise_y, no_noise_y


class DatasetEnv(BanditEnvBase):
    """ Environment based on a real dataset. """

    def __init__(self, T, N, K, d, dirname, sigma=0.0, seed=None):
        """Init."""
        self.dirname = dirname

        self.d = d
        self.K = K
        self.N = N

        self.T = T
        self.t = 1

        self.r_t = dict()
        self.s_t = dict()
        self.best_s_t = dict()
        self.worst_s_t = dict()
        self.R_t = dict()
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
            all_rewards = [self.get_no_noise_y(id, k) for k in range(self.K)]

            self.best_reward[f"agent_{i}"] = np.max(all_rewards)
            self.best_arm[f"agent_{i}"] = np.argmax(all_rewards)
            self.worst_reward[f"agent_{i}"] = np.min(all_rewards)

    def get_no_noise_y(self, agent_id, arm_id):
        """Get the reward for agent `agent_id` and arm `arm_id`."""
        filter_entry = ((self.data.agent_id == agent_id)
                        & (self.data.arm_id == arm_id))
        if not self.data[filter_entry].empty:
            return float(self.data[filter_entry].reward)
        else:
            raise ValueError(f"DataFrame does not have reward for 'agent' = "
                             f"{agent_id} and 'arm' = {arm_id}.")

    def update_agent_stats(self, name_agent, y, no_noise_y):
        """Update all statistic as listed in __init__ doc."""

        y_max = self.best_reward[name_agent]
        y_min = self.worst_reward[name_agent]

        self._update_agent_stats(name_agent, y, no_noise_y, y_max, y_min)

    def compute_reward(self, agent_name, k):
        id_ = self.agent_i_to_env_agent_i[agent_name]

        no_noise_y = self.get_no_noise_y(id_, k)
        noise = float(self.sigma * self.rng.randn())

        return noise + no_noise_y, no_noise_y


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
