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
    def __init__(self, arms, lbda=1.0, seed=None):
        """Init."""
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

    def _update_shared(self, last_k_or_arm, last_r, t):
        """Update A and b from observation.

        Parameters
        ----------
        """
        if isinstance(last_k_or_arm, (int, np.integer)):
            last_x_k = self.arms._arms[last_k_or_arm].reshape((self.d, 1))

        else:
            last_x_k = last_k_or_arm.reshape((self.d, 1))

        self.A += last_x_k.dot(last_x_k.T)
        self.b += last_x_k * last_r

        self.inv_A = _fast_inv_sherman_morrison(self.inv_A, last_x_k)
        self.theta_hat = self.inv_A.dot(self.b)

    def _update_local(self, last_k_or_arm, last_r, t):
        """Update A and b from observation.

        Parameters
        ----------
        """
        if isinstance(last_k_or_arm, (int, np.integer)):
            last_x_k = self.arms._arms[last_k_or_arm].reshape((self.d, 1))

        else:
            last_x_k = last_k_or_arm.reshape((self.d, 1))

        self.A_local += last_x_k.dot(last_x_k.T)
        self.b_local += last_x_k * last_r

        self.inv_A_local = _fast_inv_sherman_morrison(self.inv_A_local, last_x_k)  # noqa
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
