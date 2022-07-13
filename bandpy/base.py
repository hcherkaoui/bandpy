""" Define all the agents availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from .utils import check_random_state, check_actions


class BanditEnv():
    """ Virtual class of a Bandit environment. """

    def __init__(self, T, seed=None):
        """Init."""
        self.T = T
        self.t = 0
        self.rng = check_random_state(seed)
        self.S_t = dict()

    def reset(self, seed=np.NaN):
        self.t = 0
        if not np.isnan(seed):
            self.rng = check_random_state(seed)
        self.S_t = dict()

    def step(self, actions):
        """Pull the k-th arm chosen in 'actions'."""
        actions = check_actions(actions)

        observations, rewards = dict(), dict()
        for name_agent, k in actions.items():

            r = self.compute_reward(k)

            self.update_agent_total_rewards(name_agent, r)

            observation = {'last_arm_pulled': k,
                           'last_reward': r,
                           't': self.t,
                           }

            observations[name_agent] = observation
            rewards[name_agent] = r

        info = {'n_arms': self.K,
                'best_arm': self.best_arm,
                'best_reward': self.best_reward,
                }

        self.t += 1

        done = False
        if self.T <= self.t:
            done = True

        return observations, rewards, done, info

    def update_agent_total_rewards(self, name_agent, r):
        if name_agent in self.S_t:
            self.S_t[name_agent] += r
        else:
            self.S_t[name_agent] = r

    def regret(self):
        """Expected regret."""
        if not self.S_t:
            return np.NaN

        else:
            name_agents = self.S_t.keys()
            S_t = self.S_t.values()
            regrets = [self.best_reward - R / self.t for R in S_t]
            return dict(zip(name_agents, regrets))


class Controller:
    """ Abstract class for a controller then handle multiple agents. """

    def __init__(self, N, agent_cls, agent_kwargs):
        """Init."""
        self.N = N
        self.agents = dict()
        for n in range(self.N):
            self.agents[f"agent_{n}"] = agent_cls(**agent_kwargs)

    def init(self):
        """ Make each agent pulls randomly an arm to initiliaze the simulation.
        """
        actions = dict()
        for agent_name, agent in self.agents.items():
            actions[agent_name] = agent.randomly_select_arm()
        return actions


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
        self.random_state = check_random_state(seed)

    def randomly_select_arm(self):
        """Randomly select an arm."""
        k = self.random_state.randint(self.K)
        return int(k)

    def randomly_select_one_arm(self, best_arms):
        """Randomly select a best arm in a tie case."""
        if len(best_arms) > 1:
            idx = self.random_state.randint(0, len(best_arms))
            k = best_arms[idx]

        else:
            k = best_arms

        return int(k)

    def act(self, observation, reward):
        raise NotImplementedError
