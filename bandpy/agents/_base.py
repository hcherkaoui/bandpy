""" Define all the agents availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np

from .._arms import LinearArms
from .._compils import sherman_morrison, det_rank_one_update, cholesky_rank_one_update
from .._checks import check_random_state, check_A_init


class MultiLinearAgentsBase:
    """Linear agent that handle a multi-agents setting and the sharing of
    observation while keeping a local estimation up to day.

    Parameters
    ----------
    """

    def __init__(self, arms, A_init=None, lbda=1.0, seed=None):
        """Init."""
        self.lbda = lbda

        if not isinstance(arms, LinearArms):
            self.arms = LinearArms(
                criterion_func=None,
                criterion_kwargs=None,
                criterion_grad=None,
                criterion_grad_kwargs=None,
                arms=arms,
                arm_entries=None,
            )
        else:
            self.arms = arms

        self.d = self.arms.d

        self.A_init = check_A_init(self.d, self.lbda, A_init)
        self.b_init = np.zeros((self.d, 1))

        # shared variables
        self.A = np.copy(self.A_init)
        self.b = np.copy(self.b_init)
        self.inv_A = np.linalg.inv(self.A)
        self.chol_A = np.linalg.cholesky(self.A)
        self.det_A = np.linalg.det(self.A)
        self.theta_hat = self.inv_A.dot(self.b)

        # local variables
        self.A_local = np.copy(self.A_init)
        self.b_local = np.copy(self.b_init)
        self.inv_A_local = np.linalg.inv(self.A_local)
        self.chol_A_local = np.linalg.cholesky(self.A_local)
        self.det_A_local = np.linalg.det(self.A_local)
        self.theta_hat_local = self.inv_A.dot(self.b_local)

        self.rng = check_random_state(seed)

    def _update_shared(self, last_k_or_arm, last_r):
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

        self.det_A = det_rank_one_update(self.inv_A, self.det_A, last_x_k)
        self.inv_A = sherman_morrison(self.inv_A, last_x_k)
        self.chol_A = cholesky_rank_one_update(self.chol_A, last_x_k)
        self.theta_hat = self.inv_A.dot(self.b)

    def _update_local(self, last_k_or_arm, last_r):
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

        self.det_A_local = det_rank_one_update(
            self.inv_A_local, self.det_A_local, last_x_k
        )
        self.inv_A_local = sherman_morrison(self.inv_A_local, last_x_k)
        self.chol_A_local = cholesky_rank_one_update(self.chol_A_local, last_x_k)
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
        return self.arms.select_default_arm()


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
