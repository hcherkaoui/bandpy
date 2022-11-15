""" Define all the agents availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
import numba
from scipy import optimize
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


class MultiAgents(Agent):
    """Agent that handle a multi-agents setting and the sharing of observation
    while keeping a local estimation up to day.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, arms, lbda=1.0, te=10, seed=None):
        """Init."""

        self.arms = arms
        self.d = len(arms[0])  # dimension of the problem

        self.te = te  # frequencie of exacte inv_A computation

        self.lbda = lbda

        self.A = np.eye(self.d)
        self.b = np.zeros((self.d, 1))
        self.inv_A = np.eye(self.d) / lbda
        self.theta_hat = np.zeros((self.d, 1))

        self.A_local = lbda * np.eye(self.d)
        self.b_local = np.zeros((self.d, 1))
        self.inv_A_local = np.eye(self.d) / lbda
        self.theta_hat_local = np.zeros((self.d, 1))

        super().__init__(K=len(self.arms), seed=seed)

    def _update_inv_A_A_b_shared(self, last_k, last_r, t):
        """Update A and b from observation."""
        inv_A_exact = t % self.te == 0

        for last_k_, last_r_ in zip(last_k, last_r):
            last_x_k_ = self.arms[last_k_].reshape((self.d, 1))
            self.A += last_x_k_.dot(last_x_k_.T)
            self.b += last_x_k_ * last_r_
            if not inv_A_exact:
                self.inv_A = _fast_inv_sherman_morrison(self.inv_A, last_x_k_)

        if inv_A_exact:
            self.inv_A = np.linalg.inv(self.A)

    def _update_inv_A_A_b_local(self, last_k, last_r, t):
        """Update A and b from observation."""
        inv_A_exact = t % self.te == 0

        self.A_local_update = np.zeros((self.d, self.d))
        self.b_local_update = np.zeros((self.d, 1))

        last_x_k = self.arms[last_k].reshape((self.d, 1))

        self.A_local += last_x_k.dot(last_x_k.T)
        self.b_local += last_x_k * last_r

        if not inv_A_exact:
            self.inv_A_local = _fast_inv_sherman_morrison(self.inv_A_local,
                                                          last_x_k)
        else:
            self.inv_A_local = np.linalg.inv(self.A_local)

    def _update_all_statistics(self, observation, reward):
        """Update all statistics (local and shared). """
        # fetch and rename main variables
        last_k = observation['last_arm_pulled']
        last_r = reward
        t = observation['t']

        if isinstance(last_k, tuple) & isinstance(last_r, tuple):
            last_k_local, last_k_shared = last_k
            last_r_local, last_r_shared = last_r
            self._update_inv_A_A_b_shared(last_k_shared, last_r_shared, t)
            self._update_inv_A_A_b_local(last_k_local, last_r_local, t)

        else:
            self._update_inv_A_A_b_shared([last_k], [last_r], t)
            self._update_inv_A_A_b_local(last_k, last_r, t)

        self.theta_hat = self.inv_A.dot(self.b)
        self.theta_hat_local = self.inv_A_local.dot(self.b_local)

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        mean_reward_per_arms = [float(self.theta_hat.T.dot(self.arms[k]))
                                for k in range(self.K)]
        return np.argmax(mean_reward_per_arms)


class LinUniform(MultiAgents):
    """ Uniform agent for linear bandit.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def act(self, observation, reward):
        """Select an arm."""
        self._update_all_statistics(observation, reward)
        return self.randomly_select_arm()


class LinUCB(MultiAgents):
    """ Linear Upper confidence bound class to define the UCB algorithm.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    alpha: float, confidence probability parameter
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, arms, alpha, te=10, seed=None):
        """Init."""
        self.alpha = alpha

        super().__init__(arms=arms, te=te, seed=seed)

    def act(self, observation, reward):
        """Select an arm."""
        self._update_all_statistics(observation, reward)

        # arm selection
        uu = []
        for x_k in self.arms:
            u = self.theta_hat.T.dot(x_k)
            u += self.alpha * np.sqrt(x_k.T.dot(self.inv_A).dot(x_k))
            uu.append(float(u))
        k = np.argmax(uu)
        return k


class EOptimalDesign(MultiAgents):
    """ E-(trace) optimal design algorithm.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, arms, eps=1e-10, te=10, seed=None):
        """Init."""
        self.eps = eps

        super().__init__(arms=arms, te=te, seed=seed)

        self.p = self._min_f()

    def _f(self, p):
        """Objective function."""
        A = np.zeros((self.d, self.d), dtype=float)
        l_xk_xkT = [x_k.reshape((self.d, 1)).dot(x_k.reshape((1, self.d)))
                    for x_k in self.arms]
        for p_i, xk_xkT_i in zip(p, l_xk_xkT):
            A += p_i * xk_xkT_i
        return np.linalg.norm(np.linalg.pinv(A))

    def _g_1(self, p):
        return p

    def _g_2(self, p):
        return np.sum(p) - 1.0

    def _min_f(self):
        """A-optimal design planning function."""
        mu_0 = np.array([1.0 / len(self.arms)] * len(self.arms))
        constraints = [{'type': 'ineq', 'fun': self._g_1},
                       {'type': 'eq', 'fun': self._g_2}]
        p = optimize.minimize(self._f, x0=mu_0, constraints=constraints).x
        assert not any(p < -self.eps), f"non-negative constraint violated: {p}"
        p[p < 0.0] = 0.0
        return p

    def act(self, observation, reward):
        """Select an arm."""
        self._update_all_statistics(observation, reward)

        return self.rng.choice(np.arange(self.K), p=self.p)


class GreedyLinGapE(MultiAgents):
    """ Linear Gap-based Exploration class to define the LinGapE algorithm.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    alpha: float, confidence probability parameter
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, arms, epsilon, delta, R, S, lbda, te=10, seed=None):
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

        self.epsilon = epsilon
        self.delta = delta
        self.R = R
        self.S = S
        self.lbda = lbda

        super().__init__(arms=arms, te=te, seed=seed)

    def act(self, observation, reward):
        """Select an arm."""

        self._update_all_statistics(observation, reward)

        if observation['t'] < self.K:
            # arm selection
            k = observation['t'] % self.K

        else:
            # update main statistic variables
            C = np.sqrt(np.linalg.det(self.A))
            C /= (self.delta * self.lbda ** (self.d/2.0))
            C = np.sqrt(2 * np.log(C))
            C = float(C * self.R + np.sqrt(self.lbda) * self.S)

            # best arm
            ii = []
            for x_k in self.arms:
                x_k = x_k.reshape((self.d, 1))
                r_k = x_k.T.dot(self.theta_hat)
                ii.append(float(r_k))
            i = np.argmax(ii)
            x_i = self.arms[i].reshape((self.d, 1))

            # best and most uncertain arm
            jj = []
            for x_k in self.arms:
                x_k = x_k.reshape((self.d, 1))
                gap_ki = x_k - x_i
                r_hat = gap_ki.T.dot(self.theta_hat)
                ucb = np.sqrt(gap_ki.T.dot(self.inv_A).dot(gap_ki))
                jj.append(float(r_hat) + float(ucb) * C)
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
                inv_A_x_k = _fast_inv_sherman_morrison(self.inv_A, x_k)
                gap_ij = x_i - x_j
                a = np.sqrt(gap_ij.T.dot(inv_A_x_k).dot(gap_ij))
                aa.append(float(a))
            k = np.argmax(aa)

        return k
