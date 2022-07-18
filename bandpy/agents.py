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

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        mean_reward_per_arms = [np.mean(self.reward_per_arms[k])
                                for k in range(self.K)]
        filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
        best_arms = np.arange(self.K)[filter]
        return self.randomly_select_one_arm(best_arms)

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
            filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
            best_arms = np.arange(self.K)[filter]
            k = self.randomly_select_one_arm(best_arms)

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
        return self.randomly_select_one_arm(best_arms)

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

        best_arms = np.arange(self.K)[np.max(uu) == uu]
        k = self.randomly_select_one_arm(best_arms)

        return k


class LinUCB(Agent):
    """ Linear Upper confidence bound class to define the UCB algorithm.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    alpha: float, confidence probability parameter
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, arms, alpha, seed=None):
        """Init."""

        if not (alpha > 0.0):
            raise ValueError(f"'delta' should be positive, got {alpha}")

        self.arms = arms
        self.d = len(arms[0])  # dimension of the problem

        self.alpha = alpha
        self.A = np.eye(self.d)
        self.b = np.zeros((self.d, 1))

        self.inv_A = np.linalg.pinv(self.A)
        self.theta_hat = self.inv_A.dot(self.b)

        super().__init__(K=len(self.arms), seed=seed)

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        mean_reward_per_arms = [float(self.theta_hat.T.dot(self.arms[k]))
                                for k in range(self.K)]
        filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
        best_arms = np.arange(self.K)[filter]
        return self.randomly_select_one_arm(best_arms)

    def act(self, observation, reward):
        """Select an arm."""

        # fetch and rename main variables
        last_k = observation['last_arm_pulled']
        last_r = reward

        # update main statistic variables
        last_x_k = self.arms[last_k].reshape((self.d, 1))

        self.A += last_x_k.dot(last_x_k.T)
        self.b += last_x_k * last_r

        self.inv_A = np.linalg.pinv(self.A)
        self.theta_hat = self.inv_A.dot(self.b)

        # arm selection
        uu = []
        for x_k in self.arms:
            u = self.theta_hat.T.dot(x_k)
            u += self.alpha * np.sqrt(x_k.T.dot(self.inv_A).dot(x_k))
            uu.append(float(u))

        best_arms = np.arange(self.K)[np.max(uu) == uu]
        k = self.randomly_select_one_arm(best_arms)

        return k


class GreedyLinGapE(Agent):
    """ Linear Gap-based Exploration class to define the LinGapE algorithm.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    alpha: float, confidence probability parameter
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, arms, epsilon, delta, R, S, lbda, seed=None):
        """Init."""

        if not (epsilon >= 0.0):
            raise ValueError(f"'epsilon' should be positive, got {epsilon}")

        if not ((delta > 0.0) and (delta < 1.0)):
            raise ValueError(f"'delta' should belong to ]0, 1[, got {delta}")

        if not (R > 0.0):
            raise ValueError(f"'R' should be positive, got {R}")

        if not (S > 0.0):
            raise ValueError(f"'S' should be positive, got {S}")

        if not (lbda > 0.0):
            raise ValueError(f"'lambda' should be positive, got {lbda}")

        self.done = False
        self.estimated_best_arm = None

        self.arms = arms
        self.d = len(arms[0])  # dimension of the problem

        self.epsilon = epsilon
        self.delta = delta
        self.R = R
        self.S = S
        self.lbda = lbda

        self.epsilon = epsilon
        self.A = self.lbda * np.eye(self.d)
        self.b = np.zeros((self.d, 1))

        self.inv_A = np.linalg.pinv(self.A)
        self.theta_hat = self.inv_A.dot(self.b)

        super().__init__(K=len(self.arms), seed=seed)

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        if self.estimated_best_arm is not None:
            return self.estimated_best_arm

        else:
            mean_reward_per_arms = [float(self.theta_hat.T.dot(self.arms[k]))
                                    for k in range(self.K)]
            filter = np.max(mean_reward_per_arms) == mean_reward_per_arms
            best_arms = np.arange(self.K)[filter]
            return self.randomly_select_one_arm(best_arms)

    def act(self, observation, reward):
        """Select an arm."""

        # fetch and rename main variables
        t = observation['t']
        last_k = observation['last_arm_pulled']
        last_r = reward

        # update main common statistic variables
        last_x_k = self.arms[last_k].reshape((self.d, 1))
        self.A += last_x_k.dot(last_x_k.T)
        self.b += last_x_k * last_r

        if t < self.K:
            # arm selection
            k = t % self.K

        else:
            # update main statistic variables
            C = np.sqrt(np.linalg.det(self.A))
            C /= (self.delta * self.lbda ** (self.d/2.0))
            C = np.sqrt(2 * np.log(C))
            C = float(C * self.R + np.sqrt(self.lbda) * self.S)

            # estimate theta
            self.inv_A = np.linalg.pinv(self.A)
            self.theta_hat = self.inv_A.dot(self.b)

            # best arm
            ii = []
            for x_k in self.arms:
                x_k = x_k.reshape((self.d, 1))
                r_k = x_k.T.dot(self.theta_hat)
                ii.append(float(r_k))
            i = np.argmax(ii)
            x_i = self.arms[i].reshape((self.d, 1))

            # worst and most incertain arm
            jj = []
            for x_k in self.arms:
                x_k = x_k.reshape((self.d, 1))
                gap_ki = x_k - x_i
                diff_r = gap_ki.T.dot(self.theta_hat)
                ub = np.sqrt(gap_ki.T.dot(self.inv_A).dot(gap_ki))
                jj.append(float(diff_r) + float(ub) * C)
            j = np.argmax(jj)
            x_j = self.arms[j].reshape((self.d, 1))
            B = np.max(jj)

            # early-stopping
            if B <= self.epsilon:
                self.done = True  # best arm identified
                self.best_arm_hat = i

            # arm selection
            aa = []
            for x_k in self.arms:
                x_k = x_k.reshape((self.d, 1))
                inv_A_x_k = np.linalg.pinv(self.A + x_k.dot(x_k.T))
                gap_ij =  x_i - x_j
                a = np.sqrt(gap_ij.T.dot(inv_A_x_k).dot(gap_ij))
                aa.append(float(a))
            best_arms = np.arange(self.K)[np.min(aa) == aa]
            k = self.randomly_select_one_arm(best_arms)

        return k
