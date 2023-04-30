""" Define all the MAB agents availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np

from ._base import SingleMABAgentBase


class FollowTheLeader(SingleMABAgentBase):
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
        if not.

        Parameters
        ----------
        """
        mean_reward_per_arms = [np.mean(self.reward_per_arms[k]) for k in range(self.K)]
        return np.argmax(mean_reward_per_arms)

    def update_local(self, last_k, last_r):
        """Update local variables.

        Parameters
        ----------
        """
        self.reward_per_arms[last_k].append(last_r)

    def update_shared(self, last_k, last_r):
        """Update local variables.

        Parameters
        ----------
        """
        pass

    def act(self, t):
        """Select an arm.

        Parameters
        ----------
        """
        return self.best_arm


class Uniform(SingleMABAgentBase):
    """Uniform class to define a simple agent that choose the arm
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
        if not.

        Parameters
        ----------
        """
        return None

    def update_local(self, last_k, last_r):
        """Update local variables.

        Parameters
        ----------
        """
        pass

    def update_shared(self, last_k, last_r):
        """Update local variables.

        Parameters
        ----------
        """
        pass

    def act(self, t):
        """Select an arm."""
        return self.rng.randint(self.K)


class EC(SingleMABAgentBase):
    """Explore-and-Commit class to define a simple agent that randomly explore
    the arms and commit to the best estimated arm.

    Parameters
    ----------
    K : int, number of arms to consider.
    T : int, the iteration finite horizon.
    m : float, exploring time-ratio.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, Te, K, seed=None):
        """Init."""

        super().__init__(K=K, seed=seed)

        self.done = False
        self.estimated_best_arm = None

        self.Te = Te
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not.

        Parameters
        ----------
        """
        return self.estimated_best_arm

    def update_local(self, last_k, last_r):
        """Update local variables.

        Parameters
        ----------
        """
        self.reward_per_arms[last_k].append(last_r)

    def update_shared(self, last_k, last_r):
        """Update local variables.

        Parameters
        ----------
        """
        pass

    def act(self, t):
        """Select an arm.

        Parameters
        ----------
        """
        if t <= self.Te:
            k = t % self.K

        elif (t > self.Te) and not self.done:
            #  trigger once to store 'estimated_best_arm'

            mean_reward_per_arms = [
                np.mean(self.reward_per_arms[k]) for k in range(self.K)
            ]
            k = np.argmax(mean_reward_per_arms)

            self.estimated_best_arm = k

            self.done = True  # best arm identified

        else:
            k = self.estimated_best_arm

        return k


class UCB(SingleMABAgentBase):
    """Upper confidence bound class to define the UCB algorithm.

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
            raise ValueError(f"'delta' should belong to, " f"]0.0, 1.0[, got {delta}")

        self.delta = delta
        self.n_pulls_per_arms = dict([(k, 0) for k in range(self.K)])
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not.

        Parameters
        ----------
        """
        mean_reward_per_arms = [np.mean(self.reward_per_arms[k]) for k in range(self.K)]
        return np.argmax(mean_reward_per_arms)

    def update_local(self, last_k, last_r):
        """Update local variables.

        Parameters
        ----------
        """
        self.n_pulls_per_arms[last_k] += 1
        self.reward_per_arms[last_k].append(last_r)

    def update_shared(self, last_k, last_r):
        """Update local variables.

        Parameters
        ----------
        """
        pass

    def act(self, t):
        """Select an arm.

        Parameters
        ----------
        """
        uu = []
        for k in range(self.K):
            T_k = self.n_pulls_per_arms[k]
            mu_k = np.mean(self.reward_per_arms[k])
            if T_k == 0:
                uu.append(np.inf)
            else:
                uu.append(mu_k + np.sqrt(2 * np.log(1.0 / self.delta) / T_k))

        k = np.argmax(uu)

        return k
