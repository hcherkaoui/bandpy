""" Define all the agents availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from .utils import check_random_state


class FollowTheLeader:
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
        self.K = K
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])
        self.random_state = check_random_state(seed)

    def act(self, kwargs):
        """Choose the best arm observed so far."""

        observation = kwargs['observation']
        reward = kwargs['last_reward']

        self.reward_per_arms[observation['arm_pulled']].append(reward)

        mean_reward_per_arms = [np.mean(self.reward_per_arms[k])
                                for k in range(self.K)]
        filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
        best_arms = np.arange(self.K)[filter]

        if len(best_arms) > 1:  # tie case
            idx = self.random_state.randint(0, len(best_arms))
            k = best_arms[idx]
        else:
            k = int(best_arms)

        return k


class Uniform:
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
        self.K = K
        self.random_state = check_random_state(seed)

    def act(self, kwargs):
        """Choose the best arm observed so far."""
        return self.random_state.randint(self.K)


class EC:
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
        if not ((m > 0.0) and (m < 1.0)):
            raise ValueError(f"'m' (exploring time-ratio) should belong to, "
                             f"]0.0, 1.0[, got {m}")

        self.K = K
        self.Te = int(m * T)
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])
        self.random_state = check_random_state(seed)

    def act(self, kwargs):
        """Choose the best arm observed so far."""

        observation = kwargs['observation']
        reward = kwargs['last_reward']

        if observation['t'] <= self.Te:
            self.reward_per_arms[observation['arm_pulled']].append(reward)
            k = observation['t'] % self.K

        else:
            mean_reward_per_arms = [np.mean(self.reward_per_arms[k])
                                    for k in range(self.K)]
            filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
            best_arms = np.arange(self.K)[filter]

            if len(best_arms) > 1:  # tie case
                idx = self.random_state.randint(0, len(best_arms))
                k = best_arms[idx]
            else:
                k = int(best_arms)

        return k


class UCB:
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
        if not ((delta > 0.0) and (delta < 1.0)):
            raise ValueError(f"'delta' should belong to, "
                             f"]0.0, 1.0[, got {delta}")

        self.K = K
        self.delta = delta
        self.n_pulls_per_arms = dict([(k, 0) for k in range(self.K)])
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])
        self.random_state = check_random_state(seed)

    def act(self, kwargs):
        """Choose the best arm observed so far."""
        observation = kwargs['observation']
        reward = kwargs['last_reward']

        self.n_pulls_per_arms[observation['arm_pulled']] += 1
        self.reward_per_arms[observation['arm_pulled']].append(reward)

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

        if len(best_arms) > 1:  # tie case
            idx = self.random_state.randint(0, len(best_arms))
            k = best_arms[idx]
        else:
            k = int(best_arms)

        return k
