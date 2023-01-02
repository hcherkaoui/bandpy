""" Define all the agents availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np

from ..utils import tolerant_mean
from .._checks import check_random_state, check_actions


class BanditEnvBase():
    """ Virtual class of a Bandit environment.

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
        self.s_t = dict()
        self.S_t = dict()
        self.S_T = dict()
        self.no_noise_s_t = dict()
        self.no_noise_S_t = dict()
        self.no_noise_S_T = dict()

        self.best_s_t = dict()
        self.worst_s_t = dict()
        self.best_S_t = dict()
        self.worst_S_t = dict()
        self.best_S_T = dict()
        self.worst_S_T = dict()

        self.r_t = dict()
        self.R_t = dict()
        self.R_T = dict()
        self.no_noise_r_t = dict()
        self.no_noise_R_t = dict()
        self.no_noise_R_T = dict()

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

    def step(self, actions):
        """Pull the k-th arm chosen in 'actions'.

        Parameters
        ----------
        """
        actions = check_actions(actions)

        observations, rewards = dict(), dict()
        for name_agent, k_or_arm in actions.items():

            y, no_noise_y = self.compute_reward(name_agent, k_or_arm)

            self.update_agent_stats(name_agent, y, no_noise_y)

            observation = {'last_arm_pulled': k_or_arm,
                           'last_reward': y,
                           'last_no_noise_reward': no_noise_y,
                           't': self.t,
                           }

            observations[name_agent] = observation
            rewards[name_agent] = y

        info = {'best_arm': self.best_arm,
                'best_reward': self.best_reward,
                'seed': self.seed,
                }

        self.t += 1

        done = False
        if self.T < self.t:
            done = True

        return observations, rewards, done, info

    def _update_agent_stats(self, name_agent, y, no_noise_y, y_max, y_min):
        """Update all statistic as listed in __init__ doc.

        Parameters
        ----------
        """

        # only check self.S_t since there are -all- updated together # noqa
        if name_agent in self.S_t:
            self.s_t[name_agent].append(y)
            self.S_t[name_agent].append(self.S_t[name_agent][-1] + y)
            self.S_T[name_agent] = self.S_t[name_agent][-1]
            self.no_noise_s_t[name_agent].append(no_noise_y)
            self.no_noise_S_t[name_agent].append(self.no_noise_S_t[name_agent][-1] + no_noise_y)  # noqa
            self.no_noise_S_T[name_agent] = self.no_noise_S_t[name_agent][-1]

            self.best_s_t[name_agent].append(y_max)
            self.worst_s_t[name_agent].append(y_min)
            self.best_S_t[name_agent].append(self.best_S_t[name_agent][-1] + y_max)  # noqa
            self.worst_S_t[name_agent].append(self.worst_S_t[name_agent][-1] + y_min)  # noqa
            self.best_S_T[name_agent] = self.best_S_t[name_agent][-1]
            self.worst_S_T[name_agent] = self.worst_S_t[name_agent][-1]

            self.r_t[name_agent].append(y_max - y)
            self.R_t[name_agent].append(self.R_t[name_agent][-1] + y_max - y)
            self.R_T[name_agent] = self.R_t[name_agent][-1]
            self.no_noise_r_t[name_agent].append(y_max - no_noise_y)
            self.no_noise_R_t[name_agent].append(self.no_noise_R_t[name_agent][-1] + y_max - no_noise_y)  # noqa
            self.no_noise_R_T[name_agent] = self.no_noise_R_t[name_agent][-1]

        else:
            self.s_t[name_agent] = [y]
            self.S_t[name_agent] = [y]
            self.S_T[name_agent] = self.S_t[name_agent][-1]
            self.no_noise_s_t[name_agent] = [no_noise_y]
            self.no_noise_S_t[name_agent] = [no_noise_y]
            self.no_noise_S_T[name_agent] = self.no_noise_S_t[name_agent][-1]

            self.best_s_t[name_agent] = [y_max]
            self.worst_s_t[name_agent] = [y_min]
            self.best_S_t[name_agent] = [y_max]
            self.worst_S_t[name_agent] = [y_min]
            self.best_S_T[name_agent] = self.best_S_t[name_agent][-1]
            self.worst_S_T[name_agent] = self.worst_S_t[name_agent][-1]

            self.r_t[name_agent] = [y_max - y]
            self.R_t[name_agent] = [y_max - y]
            self.R_T[name_agent] = self.R_t[name_agent][-1]
            self.no_noise_r_t[name_agent] = [y_max - no_noise_y]
            self.no_noise_R_t[name_agent] = [y_max - no_noise_y]  # noqa
            self.no_noise_R_T[name_agent] = self.no_noise_R_t[name_agent][-1]

    def mean_instantaneous_reward(self):
        """Return the averaged (on the network) instantaneous reward (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.s_t.values()))

    def mean_instantaneous_no_noise_reward(self):
        """Return the averaged (on the network) instantaneous reward -without-
        nosie (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.no_noise_s_t.values()))

    def mean_instantaneous_best_reward(self):
        """Return the averaged (on the network) best instantaneous reward
        (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.best_s_t.values()))

    def mean_instantaneous_worst_reward(self):
        """Return the averaged (on the network) worst instantaneous reward
        (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.worst_s_t.values()))

    def mean_instantaneous_regret(self):
        """Return the averaged (on the network) instantaneous regret (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.r_t.values()))

    def mean_instantaneous_no_noise_regret(self):
        """Return the averaged (on the network) instantaneous regret -without-
        noise (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.no_noise_r_t.values()))

    def mean_cumulative_regret(self):
        """Return the averaged (on the network) cumulative regret (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.R_t.values()))

    def mean_cumulative_no_noise_regret(self):
        """Return the averaged (on the network) cumulative regret -without
        nosie- (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.no_noise_R_t.values()))

    def mean_cumulative_regret_last_value(self):
        """Return the averaged (on the network) cumulative regret last value
        (float).

        Parameters
        ----------
        """
        return np.mean(list(self.R_T.values()))

    def mean_cumulative_no_noise_regret_last_value(self):
        """Return the averaged (on the network) cumulative regret -without
        nosie- last value (float).

        Parameters
        ----------
        """
        return np.mean(list(self.no_noise_R_T.values()))

    def mean_cumulative_reward(self):
        """Return the network averaged cumulative reward (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.S_t.values()))

    def mean_cumulative_no_noise_reward(self):
        """Return the network averaged cumulative reward -without noise-
        (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.no_noise_S_t.values()))

    def mean_cumulative_reward_last_value(self):
        """Return the network averaged cumulative reward last value (float).

        Parameters
        ----------
        """
        return np.mean(list(self.S_T.values()))

    def mean_cumulative_no_noise_reward_last_value(self):
        """Return the network averaged cumulative reward -without noise- last
        value (float).

        Parameters
        ----------
        """
        return np.mean(list(self.no_noise_S_T.values()))

    def mean_cumulative_best_reward(self):
        """Return the network averaged best cumulative reward (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.best_S_t.values()))

    def mean_cumulative_worst_reward(self):
        """Return the network averaged worst cumulative reward (array).

        Parameters
        ----------
        """
        return tolerant_mean(list(self.worst_S_t.values()))
