""" Define all the agents availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np

from .._compils import _fast_inv_sherman_morrison
from .._checks import check_random_state


class MultiLinearAgentsBase:
    """Linear agent that handle a multi-agents setting and the sharing of
    observation while keeping a local estimation up to day.

    Parameters
    ----------
    """
    def __init__(self, arms, lbda=1.0, te=10, seed=None):
        """Init."""
        self.te = te  # frequencie of exacte inv_A computation

        self.lbda = lbda

        self.arms = arms
        self.d = self.arms.d

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
        """Update A and b from observation.

        Parameters
        ----------
        """
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
        """Update A and b from observation.

        Parameters
        ----------
        """
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
        """Update all statistics (local and shared).

        Parameters
        ----------
        """
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
        if not.

        Parameters
        ----------
        """
        return self.arms.best_arm(self.theta_hat)

    def select_default_arm(self):
        """Select the 'default arm'.

        Parameters
        ----------
        """
        return self.arms.select_default_arm(arm_entries=None)


class SingleMABAgentBase:
    """MAB agent that only handle a single-agents setting.

    Parameters
    ----------
    """
    def __init__(self, K, seed=None):
        """Init."""
        self.K = K
        self.rng = check_random_state(seed)

    def select_default_arm(self):
        """Select the 'default arm'."""
        return 0
