""" Define all the bandit environments available in bandpy.
"""
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

    def observe_env(self, k, reward):
        """Store the reward collected from the k arm."""
        self.reward_per_arms[k].append(reward)
