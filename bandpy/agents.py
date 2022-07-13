""" Define all the agents availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from .base import Agent


class FollowTheLeader(Agent):
    """FollowTheLeader class to define a simple agent that choose the best arm
    observed so far.

    Parameters
    ----------
    K : int, number of arms to consider.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, K, seed=None):
        """Init."""
        super().__init__(K=K, seed=seed)
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])

    def act(self, observation, reward):
        """Choose the best arm observed so far."""

        # fetch and rename main variables
        last_k = observation['last_arm_pulled']
        last_r = reward

        # update main statistic variables
        self.reward_per_arms[last_k].append(last_r)

        # arm selection
        mean_reward_per_arms = [np.mean(self.reward_per_arms[k])
                                for k in range(self.K)]
        filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
        best_arms = np.arange(self.K)[filter]

        return self.randomly_select_one_arm(best_arms)


class Uniform(Agent):
    """ Uniform class to define a simple agent that choose the arm
    randomly.

    Parameters
    ----------
    K : int, number of arms to consider.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """
    def __init__(self, K, seed=None):
        """Init."""
        super().__init__(K=K, seed=seed)

    def act(self, observation, reward):
        """Choose the best arm observed so far."""
        return self.randomly_select_arm()


class EC(Agent):
    """ Explore-and-Commit class to define a simple agent that randomly explore
    the arms and commit to the best estimated arm.

    Parameters
    ----------
    K : int, number of arms to consider.
    T : int, the iteration finite horizon.
    m : float, exploring time-ratio.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, K, T, m=0.5, seed=None):
        """Init."""

        super().__init__(K=K, seed=seed)

        if not ((m > 0.0) and (m < 1.0)):
            raise ValueError(f"'m' (exploring time-ratio) should belong to, "
                             f"]0.0, 1.0[, got {m}")

        self.Te = int(m * T)
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])

    def act(self, observation, reward):
        """Choose the best arm observed so far."""

        # fetch and rename main variables
        t = observation['t']
        last_k = observation['last_arm_pulled']
        last_r = reward

        # arm selection
        if t <= self.Te:
            self.reward_per_arms[last_k].append(last_r)
            k = t % self.K
        else:
            mean_reward_per_arms = [np.mean(self.reward_per_arms[k])
                                    for k in range(self.K)]
            filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
            best_arms = np.arange(self.K)[filter]

            k = self.randomly_select_one_arm(best_arms)

        return k


class UCB(Agent):
    """ Upper confidence bound class to define the UCB algorithm.

    Parameters
    ----------
    K : int, number of arms to consider.
    delta: float, confidence probability parameter
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, K, delta, seed=None):
        """Init."""

        super().__init__(K=K, seed=seed)

        if not ((delta > 0.0) and (delta < 1.0)):
            raise ValueError(f"'delta' should belong to, "
                             f"]0.0, 1.0[, got {delta}")

        self.delta = delta
        self.n_pulls_per_arms = dict([(k, 0) for k in range(self.K)])
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])

    def act(self, observation, reward):
        """Choose the best arm observed so far."""

        # fetch and rename main variables
        last_k = observation['last_arm_pulled']
        last_r = reward

        # update main statistic variables
        self.n_pulls_per_arms[last_k] += 1
        self.reward_per_arms[last_k].append(last_r)

        # arm selection
        u = []
        for k in range(self.K):
            T_k = self.n_pulls_per_arms[k]
            mu_k = np.mean(self.reward_per_arms[k])
            if T_k == 0:
                u.append(np.inf)
            else:
                u.append(mu_k + np.sqrt(2 * np.log(1.0/self.delta) / T_k))
        filter = np.max(u) == u
        best_arms = np.arange(self.K)[filter]
        k = self.randomly_select_one_arm(best_arms)

        return k
