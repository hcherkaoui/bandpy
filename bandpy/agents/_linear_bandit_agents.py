""" Define all the linear agents availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from scipy import optimize

from ._base import MultiLinearAgentsBase
from .._criterions import _f_ucb, f_neg_ucb, grad_neg_ucb
from .._arms import LinearArms, QuadraticArms, arm_to_quadratic_arm
from .._compils import sherman_morrison
from ..utils import get_d


class LinUniform(MultiLinearAgentsBase):
    """ Uniform agent for linear bandit.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """
    def __init__(self, arms=None, arm_entries=None, lbda=1.0, seed=None):
        """Init."""
        # init internal arms class
        self.arms = LinearArms(criterion_func=None,
                               criterion_kwargs=None,
                               criterion_grad=None,
                               criterion_grad_kwargs=None,
                               arms=arms, arm_entries=arm_entries)

        # init last variables
        self.K = self.arms.K

        # init internal variables (inv_A, A, ...)
        super().__init__(arms=self.arms, lbda=lbda, seed=seed)

    def select_default_arm(self):
        """Select the 'default arm'."""
        return self.arms.select_default_arm()

    def update_local(self, last_k_or_arm, last_r):
        """Update local variables."""
        self._update_local(last_k_or_arm, last_r)

    def update_shared(self, last_k_or_arm, last_r):
        """Update shared variables."""
        self._update_shared(last_k_or_arm, last_r)

    def act(self, t):
        """Select an arm."""
        if self.arms.return_arm_index:
            return self.rng.randint(self.K)

        else:
            random_arm = []
            for entry_vals in self.arms._arm_entries.values():
                random_arm.append(self.rng.choice(entry_vals))
            return np.array(random_arm, dtype=float).reshape((self.d, 1))


class LinUCB(MultiLinearAgentsBase):
    """ Linear Upper confidence bound class to define the UCB algorithm.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    alpha: float, confidence probability parameter
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """
    def __init__(self, alpha, arms=None, arm_entries=None,
                 lbda=1.0, seed=None):
        """Init."""
        d = get_d(arms=arms, arm_entries=arm_entries)

        # init internal arms class
        criterion_kwargs = dict(alpha=alpha, t=0, inv_A=np.eye(d) / lbda)
        criterion_grad_kwargs = dict(alpha=alpha, t=0, inv_A=np.eye(d) / lbda)
        self.arms = LinearArms(criterion_func=f_neg_ucb,
                               criterion_kwargs=criterion_kwargs,
                               criterion_grad=grad_neg_ucb,
                               criterion_grad_kwargs=criterion_grad_kwargs,
                               arms=arms, arm_entries=arm_entries)

        # init last variables
        self.alpha = alpha
        self.K = self.arms.K

        # init internal variables (inv_A, A, ...)
        super().__init__(arms=self.arms, lbda=lbda, seed=seed)

    def select_default_arm(self):
        """Select the 'default arm'."""
        return self.arms.select_default_arm()

    def update_local(self, last_k_or_arm, last_r):
        """Update local variables."""
        self._update_local(last_k_or_arm, last_r)

    def update_shared(self, last_k_or_arm, last_r):
        """Update shared variables."""
        self._update_shared(last_k_or_arm, last_r)

    def act(self, t):
        """Select an arm."""
        kwargs = dict(alpha=self.alpha, t=t, inv_A=self.inv_A)  # shared inv_A

        selected_k_or_arm = self.arms.select_arm(
                                self.theta_hat,  # shared theta
                                criterion_kwargs=kwargs,
                                criterion_grad_kwargs=kwargs)

        return selected_k_or_arm


class QuadUCB(MultiLinearAgentsBase):
    """ Quadratic Upper confidence bound class to define the UCB algorithm.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    alpha: float, confidence probability parameter
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """
    def __init__(self, alpha, arms=None, arm_entries=None,
                 lbda=1.0, seed=None):
        """Init."""
        d = get_d(arms=arms, arm_entries=arm_entries)

        # init internal arms class
        criterion_kwargs = dict(alpha=alpha, t=0, inv_A=np.eye(2 * d + 1) / lbda)  # noqa
        criterion_grad_kwargs = dict(alpha=alpha, t=0, inv_A=np.eye(2 * d + 1) / lbda)  # noqa
        self.arms = QuadraticArms(criterion_func=f_neg_ucb,
                                  criterion_kwargs=criterion_kwargs,
                                  criterion_grad=grad_neg_ucb,
                                  criterion_grad_kwargs=criterion_grad_kwargs,
                                  arms=arms, arm_entries=arm_entries)

        # init last variables
        self.alpha = alpha
        self.K = self.arms.K

        # init internal variables (inv_A, A, ...)
        super().__init__(arms=self.arms, lbda=lbda, seed=seed)

    def select_default_arm(self):
        """Select the 'default arm'."""
        return self.arms.select_default_arm()

    def update_local(self, last_k_or_arm, last_r):
        """Update local variables after the quadratic formating."""

        if isinstance(last_k_or_arm, np.ndarray):
            last_k_or_arm = arm_to_quadratic_arm(last_k_or_arm)

        self._update_local(last_k_or_arm, last_r)

    def update_shared(self, last_k_or_arm, last_r):
        """Update shared variables after the quadratic formating."""

        if isinstance(last_k_or_arm, np.ndarray):
            last_k_or_arm = arm_to_quadratic_arm(last_k_or_arm)

        self._update_shared(last_k_or_arm, last_r)

    def act(self, t):
        """Select an arm."""
        kwargs = dict(alpha=self.alpha, t=t, inv_A=self.inv_A)  # shared inv_A

        selected_k_or_arm = self.arms.select_arm(
                                self.theta_hat,  # shared theta
                                criterion_kwargs=kwargs,
                                criterion_grad_kwargs=kwargs)

        return selected_k_or_arm


# TODO make it handle the arm_entries case
class EOptimalDesign(MultiLinearAgentsBase):
    """ E-(trace) optimal design algorithm.

    Parameters
    ----------
    arms : list of np.array, list of arms.
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    """

    def __init__(self, arms, eps=1e-10, lbda=1.0, seed=None):
        """Init."""
        self.eps = eps

        # XXX re-use the LinearArms for now
        self.arms = LinearArms(criterion_func=f_neg_ucb,
                               criterion_kwargs=dict(),
                               criterion_grad=grad_neg_ucb,
                               criterion_grad_kwargs=dict(),
                               arms=arms, arm_entries=None)

        super().__init__(arms=self.arms, lbda=lbda, seed=seed)

        self.p = self._min_f()

    def _f(self, p):
        """Objective function."""
        A = np.zeros((self.d, self.d), dtype=float)
        l_xk_xkT = [x_k.reshape((self.d, 1)).dot(x_k.reshape((1, self.d)))
                    for x_k in self.arms._arms]
        for p_i, xk_xkT_i in zip(p, l_xk_xkT):
            A += p_i * xk_xkT_i
        return np.linalg.norm(np.linalg.pinv(A))

    def _g_1(self, p):
        return p

    def _g_2(self, p):
        return np.sum(p) - 1.0

    def _min_f(self):
        """A-optimal design planning function."""
        mu_0 = np.array([1.0 / len(self.arms._arms)] * len(self.arms._arms))
        constraints = [{'type': 'ineq', 'fun': self._g_1},
                       {'type': 'eq', 'fun': self._g_2}]
        p = optimize.minimize(self._f, x0=mu_0, constraints=constraints).x
        assert not any(p < -self.eps), f"non-negative constraint violated: {p}"
        p[p < 0.0] = 0.0
        return p

    def update_local(self, last_k_or_arm, last_r):
        """Update local variables."""
        self._update_local(last_k_or_arm, last_r)

    def update_shared(self, last_k_or_arm, last_r):
        """Update shared variables."""
        self._update_shared(last_k_or_arm, last_r)

    def select_default_arm(self):
        """Select the 'default arm'."""
        return self.arms.select_default_arm()

    def act(self, t):
        """Select an arm."""
        return self.rng.choice(np.arange(self.arms.K), p=self.p)


# TODO make it handle the arm_entries case
class GreedyLinGapE(MultiLinearAgentsBase):
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

        self.epsilon = epsilon
        self.delta = delta
        self.R = R
        self.S = S
        self.lbda = lbda

        super().__init__(arms=arms, seed=seed)

    def update_local(self, last_k_or_arm, last_r):
        """Update local variables."""
        self._update_local(last_k_or_arm, last_r)

    def update_shared(self, last_k_or_arm, last_r):
        """Update shared variables."""
        self._update_shared(last_k_or_arm, last_r)

    def act(self, t):
        """Select an arm."""
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
            jj.append(_f_ucb(x_k - x_i, C, self.theta_hat,
                      self.inv_A_local))
        j = np.argmax(jj)
        x_j = self.arms[j].reshape((self.d, 1))
        B = np.max(jj)

        # early-stopping
        if B <= self.epsilon:
            self.done = True  # best arm identified
            self.best_arm_hat = i

        # greedy arm selection
        aa = []
        for x_k in self.arms:
            x_k = x_k.reshape((self.d, 1))
            inv_A_x_k = sherman_morrison(self.inv_A_local, x_k)
            gap_ij = x_i - x_j
            a = np.sqrt(gap_ij.T.dot(inv_A_x_k).dot(gap_ij))
            aa.append(float(a))
        k = np.argmax(aa)

        return k
