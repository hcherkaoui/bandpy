""" Define all the agents availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np

from .compils import _fast_inv_sherman_morrison
from .utils import tolerant_mean
from .checks import check_random_state, check_actions
from .arms import _select_default_arm


class BanditEnv():
    """ Virtual class of a Bandit environment. """

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
        """Reset the environment (and the randomness if seed is not NaN)."""
        # reset randomness
        self.seed = seed
        self.rng = check_random_state(self.seed)

        # re-set reward tracking
        self.init_metrics()

        # re-set time
        self.t = 1

    def step(self, actions):
        """Pull the k-th arm chosen in 'actions'."""
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

    def update_agent_stats(self, name_agent, y, no_noise_y):
        """Update all statistic as listed in __init__ doc."""

        theta_idx = self.theta_per_agent[name_agent]

        y_max = self.best_reward[theta_idx]
        y_min = self.worst_reward[theta_idx]

        self._update_agent_stats(name_agent, y, no_noise_y, y_max, y_min)

    def _update_agent_stats(self, name_agent, y, no_noise_y, y_max, y_min):
        """Update all statistic as listed in __init__ doc."""

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
        """
        return tolerant_mean(list(self.s_t.values()))

    def mean_instantaneous_no_noise_reward(self):
        """Return the averaged (on the network) instantaneous reward -without-
        nosie (array). """
        return tolerant_mean(list(self.no_noise_s_t.values()))

    def mean_instantaneous_best_reward(self):
        """Return the averaged (on the network) best instantaneous reward
        (array)."""
        return tolerant_mean(list(self.best_s_t.values()))

    def mean_instantaneous_worst_reward(self):
        """Return the averaged (on the network) worst instantaneous reward
        (array)."""
        return tolerant_mean(list(self.worst_s_t.values()))

    def mean_instantaneous_regret(self):
        """Return the averaged (on the network) instantaneous regret (array).
        """
        return tolerant_mean(list(self.r_t.values()))

    def mean_instantaneous_no_noise_regret(self):
        """Return the averaged (on the network) instantaneous regret -without-
        nosie (array)."""
        return tolerant_mean(list(self.no_noise_r_t.values()))

    def mean_cumulative_regret(self):
        """Return the averaged (on the network) cumulative regret (array)."""
        return tolerant_mean(list(self.R_t.values()))

    def mean_cumulative_no_noise_regret(self):
        """Return the averaged (on the network) cumulative regret -without
        nosie- (array)."""
        return tolerant_mean(list(self.no_noise_R_t.values()))

    def mean_cumulative_regret_last_value(self):
        """Return the averaged (on the network) cumulative regret last value
        (float)."""
        return np.mean(list(self.R_T.values()))

    def mean_cumulative_no_noise_regret_last_value(self):
        """Return the averaged (on the network) cumulative regret -without
        nosie- last value (float)."""
        return np.mean(list(self.no_noise_R_T.values()))

    def mean_cumulative_reward(self):
        """Return the network averaged cumulative reward (array)."""
        return tolerant_mean(list(self.S_t.values()))

    def mean_cumulative_no_noise_reward(self):
        """Return the network averaged cumulative reward -without noise-
        (array)."""
        return tolerant_mean(list(self.no_noise_S_t.values()))

    def mean_cumulative_reward_last_value(self):
        """Return the network averaged cumulative reward last value (float)."""
        return np.mean(list(self.S_T.values()))

    def mean_cumulative_no_noise_reward_last_value(self):
        """Return the network averaged cumulative reward -without noise- last
        value (float)."""
        return np.mean(list(self.no_noise_S_T.values()))

    def mean_cumulative_best_reward(self):
        """Return the network averaged best cumulative reward (array)."""
        return tolerant_mean(list(self.best_S_t.values()))

    def mean_cumulative_worst_reward(self):
        """Return the network averaged worst cumulative reward (array)."""
        return tolerant_mean(list(self.worst_S_t.values()))


class Controller:
    """ Abstract class for a controller then handle multiple agents. """

    def __init__(self, N, agent_cls, agent_kwargs, agent_names=None):
        """Init."""
        self.agents = dict()

        self.N = N

        if agent_names is None:
            self.agent_names = [f"agent_{i}" for i in range(self.N)]
        else:
            self.agent_names = agent_names

        for agent_name in self.agent_names:
            self.agents[agent_name] = agent_cls(**agent_kwargs)

    @property
    def best_arms(self):
        """Return for each agent the estimated best arm. """
        best_arms = dict()
        for agent_name, agent in self.agents.items():
            best_arms[agent_name] = agent.best_arm
        return best_arms

    def default_act(self):
        """ Make each agent pulls 'default' arm to init the simulation."""
        actions = dict()
        for agent_name, agent in self.agents.items():
            actions[agent_name] = self.agents[agent_name].select_default_arm()
        return actions


class MultiLinearAgents:
    """Agent that handle a multi-agents setting and the sharing of observation
    while keeping a local estimation up to day.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """
    def __init__(self, d, lbda=1.0, te=10, seed=None):
        """Init."""
        self.d = d

        self.te = te  # frequencie of exacte inv_A computation

        self.lbda = lbda

        # shared variables
        self.A = np.eye(self.d)
        self.b = np.zeros((self.d, 1))
        self.inv_A = np.eye(self.d) / lbda
        self.theta_hat = np.zeros((self.d, 1))

        # local variables
        self.A_local = lbda * np.eye(self.d)
        self.b_local = np.zeros((self.d, 1))
        self.inv_A_local = np.eye(self.d) / lbda
        self.theta_hat_local = np.zeros((self.d, 1))

        self.rng = check_random_state(seed)

    def _update_inv_A_A_b_shared(self, last_k_or_arm, last_r, t):
        """Update A and b from observation."""
        inv_A_exact = t % self.te == 0

        for last_k_or_arm_, last_r_ in zip(last_k_or_arm, last_r):

            if isinstance(last_k_or_arm_, (int, np.integer)):
                arm_ = self.arms._arms[last_k_or_arm_]
                last_x_k_ = arm_.reshape((self.d, 1))

            else:
                last_x_k_ = last_k_or_arm_.reshape((self.d, 1))

            self.A += last_x_k_.dot(last_x_k_.T)
            self.b += last_x_k_ * last_r_
            if not inv_A_exact:
                self.inv_A = _fast_inv_sherman_morrison(self.inv_A, last_x_k_)

        if inv_A_exact:
            self.inv_A = np.linalg.inv(self.A)

    def _update_inv_A_A_b_local(self, last_k_or_arm, last_r, t):
        """Update A and b from observation."""
        inv_A_exact = t % self.te == 0

        self.A_local_update = np.zeros((self.d, self.d))
        self.b_local_update = np.zeros((self.d, 1))

        if isinstance(last_k_or_arm, (int, np.integer)):
            last_x_k = self.arms._arms[last_k_or_arm].reshape((self.d, 1))

        else:
            last_x_k = last_k_or_arm.reshape((self.d, 1))

        self.A_local += last_x_k.dot(last_x_k.T)
        self.b_local += last_x_k * last_r

        if not inv_A_exact:
            inv_A_ = _fast_inv_sherman_morrison(self.inv_A_local, last_x_k)
            self.inv_A_local = inv_A_
        else:
            self.inv_A_local = np.linalg.inv(self.A_local)

    def _update_all_statistics(self, observation, reward):
        """Update all statistics (local and shared). """
        # fetch and rename main variables
        last_k_or_arm = observation['last_arm_pulled']
        last_r = reward
        t = observation['t']

        if isinstance(last_k_or_arm, tuple) and isinstance(last_r, tuple):
            last_k_or_arm_local, last_k_or_arm_shared = last_k_or_arm
            last_r_local, last_r_shared = last_r
            self._update_inv_A_A_b_shared(last_k_or_arm_shared, last_r_shared,
                                          t)
            self._update_inv_A_A_b_local(last_k_or_arm_local, last_r_local, t)

        else:
            self._update_inv_A_A_b_shared([last_k_or_arm], [last_r], t)
            self._update_inv_A_A_b_local(last_k_or_arm, last_r, t)

        self.theta_hat = self.inv_A.dot(self.b)
        self.theta_hat_local = self.inv_A_local.dot(self.b_local)

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        return self.arms.best_arm(self.theta_hat)

    def select_default_arm(self):
        """Select the 'default arm'."""
        return _select_default_arm(arm_entries=None)


class Agent:
    """ Abstract class for an agent.

    Parameters
    ----------
    K : int, number of arms to consider.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, K, seed=None):
        """Init."""
        self.K = K
        self.rng = check_random_state(seed)

    def randomly_select_arm(self):
        """Randomly select an arm."""
        k = self.rng.randint(self.K)
        return int(k)

    def randomly_select_one_arm_from_best_arms(self, best_arms):
        """Randomly select a best arm in a tie case."""
        if len(best_arms) > 1:
            idx = self.rng.randint(0, len(best_arms))
            k = best_arms[idx]

        else:
            k = best_arms

        return int(k)

    def act(self, observation, reward):
        raise NotImplementedError
