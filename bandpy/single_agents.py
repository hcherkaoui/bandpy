""" Define all the agents availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
import numba
from .base import Agent


@numba.jit((numba.float64[:, :], numba.float64[:, :]), nopython=True,
           cache=True, fastmath=True)
def _fast_inv_sherman_morrison(inv_A, x):  # pragma: no cover
    """Sherman-Morrison identity to compute the inverse of A + xxT."""
    inv_A_x = inv_A.dot(x)
    return inv_A - inv_A_x.dot(x.T.dot(inv_A)) / (1.0 + x.T.dot(inv_A_x))


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

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        mean_reward_per_arms = [np.mean(self.reward_per_arms[k])
                                for k in range(self.K)]
        return np.argmax(mean_reward_per_arms)

    def act(self, observation, reward):
        """Select an arm."""

        # fetch and rename main variables
        last_k = observation['last_arm_pulled']
        last_r = reward

        # update main statistic variables
        self.reward_per_arms[last_k].append(last_r)

        # arm selection
        return self.best_arm


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

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        return None

    def act(self, observation, reward):
        """Select an arm."""
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

        self.done = False
        self.estimated_best_arm = None

        if not ((m > 0.0) and (m < 1.0)):
            raise ValueError(f"'m' (exploring time-ratio) should belong to, "
                             f"]0.0, 1.0[, got {m}")

        self.Te = int(m * T)
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        return self.estimated_best_arm

    def act(self, observation, reward):
        """Select an arm."""

        # fetch and rename main variables
        t = observation['t']
        last_k = observation['last_arm_pulled']
        last_r = reward

        # arm selection
        if t <= self.Te:
            self.reward_per_arms[last_k].append(last_r)
            k = t % self.K

        elif (t > self.Te) and not self.done:
            #  trigger once to store 'estimated_best_arm'

            mean_reward_per_arms = [np.mean(self.reward_per_arms[k])
                                    for k in range(self.K)]
            k = np.argmax(mean_reward_per_arms)

            self.estimated_best_arm = k

            self.done = True  # best arm identified

        else:
            k = self.estimated_best_arm

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

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        mean_reward_per_arms = [np.mean(self.reward_per_arms[k])
                                for k in range(self.K)]
        filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
        best_arms = np.arange(self.K)[filter]
        return self.randomly_select_one_arm_from_best_arms(best_arms)

    def act(self, observation, reward):
        """Select an arm."""

        # fetch and rename main variables
        last_k = observation['last_arm_pulled']
        last_r = reward

        # update main statistic variables
        self.n_pulls_per_arms[last_k] += 1
        self.reward_per_arms[last_k].append(last_r)

        # arm selection
        uu = []
        for k in range(self.K):
            T_k = self.n_pulls_per_arms[k]
            mu_k = np.mean(self.reward_per_arms[k])
            if T_k == 0:
                uu.append(np.inf)
            else:
                uu.append(mu_k + np.sqrt(2 * np.log(1.0/self.delta) / T_k))

        k = np.argmax(uu)

        return k
