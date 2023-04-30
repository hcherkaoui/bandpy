""" Define all the agents availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np

from .._checks import check_random_state, check_actions


class BanditEnvBase:
    """Virtual class of a Bandit environment.

    Parameters
    ----------
    """

    def __init__(self, T, seed=None):
        """Init."""
        self.T = T
        self.t = 1

        self.seed = seed
        self.rng = check_random_state(self.seed)

        self.init_metrics()

    def init_metrics(self):
        """Init/reset all the metrics."""
        self.s_t = np.zeros((self.T,), dtype=float)
        self.S_t = np.zeros((self.T,), dtype=float)
        self.S_T = 0.0

        self.no_noise_s_t = np.zeros((self.T,), dtype=float)
        self.no_noise_S_t = np.zeros((self.T,), dtype=float)
        self.no_noise_S_T = 0.0

        self.best_s_t = np.zeros((self.T,), dtype=float)
        self.best_S_t = np.zeros((self.T,), dtype=float)
        self.best_S_T = 0.0

        self.worst_s_t = np.zeros((self.T,), dtype=float)
        self.worst_S_t = np.zeros((self.T,), dtype=float)
        self.worst_S_T = 0.0

        self.r_t = np.zeros((self.T,), dtype=float)
        self.R_t = np.zeros((self.T,), dtype=float)
        self.R_T = 0.0

        self.no_noise_r_t = np.zeros((self.T,), dtype=float)
        self.no_noise_R_t = np.zeros((self.T,), dtype=float)
        self.no_noise_R_T = 0.0

    def reset(self, seed=None):
        """Reset the environment (and the randomness if seed is not NaN).

        Parameters
        ----------
        """
        # reset randomness
        self.seed = seed
        self.rng = check_random_state(self.seed)

        # re-set reward tracking
        self.init_metrics()

        # re-set time
        self.t = 1

    def step(self, action):
        """Pull the k-th arm chosen in 'actions'.

        Parameters
        ----------
        """
        action = check_actions(action)
        name_agent, k_or_arm = next(iter(action.items()))

        y, no_noise_y = self.compute_reward(name_agent, k_or_arm)

        self.update_agent_stats(name_agent, y, no_noise_y)

        observation = {
            "last_arm_pulled": k_or_arm,
            "last_reward": y,
            "last_no_noise_reward": no_noise_y,
            "t": self.t,
        }

        observation = {name_agent: observation}

        reward = {name_agent: y}

        info = {
            "best_arm": self.best_arm,
            "best_reward": self.best_reward,
            "seed": self.seed,
        }

        self.t += 1

        done = self.T < self.t

        return observation, reward, done, info

    def _update_agent_stats(self, name_agent, y, no_noise_y, y_max, y_min):
        """Update all statistic as listed in __init__ doc.

        Parameters
        ----------
        """
        self.s_t[self.t - 1] = y
        self.S_t[self.t - 1] = self.S_t[-1] + y
        self.S_T = self.S_t[-1]

        self.no_noise_s_t[self.t - 1] = no_noise_y
        self.no_noise_S_t[self.t - 1] = self.no_noise_S_t[self.t - 2] + no_noise_y
        self.no_noise_S_T = self.no_noise_S_t[self.t - 1]

        self.best_s_t[self.t - 1] = y_max
        self.best_S_t[self.t - 1] = self.best_S_t[self.t - 2] + y_max
        self.best_S_T = self.best_S_t[self.t - 1]

        self.worst_s_t[self.t - 1] = y_min
        self.worst_S_t[self.t - 1] = self.worst_S_t[self.t - 2] + y_min
        self.worst_S_T = self.worst_S_t[self.t - 1]

        self.r_t[self.t - 1] = y_max - y
        self.R_t[self.t - 1] = self.R_t[-1] + y_max - y
        self.R_T = self.R_t[-1]

        self.no_noise_r_t[self.t - 1] = y_max - no_noise_y
        self.no_noise_R_t[self.t - 1] = (
            self.no_noise_R_t[self.t - 2] + y_max - no_noise_y
        )
        self.no_noise_R_T = self.no_noise_R_t[self.t - 1]
