""" Define the arm classes and routines. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import warnings
import numpy as np
from scipy import optimize

from .criterions import f_neg_scalar_prod, grad_neg_scalar_prod
from .utils import (arms_to_arm_entries, arm_entries_to_arms,
                    proj_on_arm_entries)
from .__init__ import MAX_K


class LinearArms:
    """Linear arms class."""

    def __init__(self, criterion_func, criterion_kwargs,
                 criterion_grad=None, criterion_grad_kwargs=None,
                 arms=None, arm_entries=None, return_arm_index=True):
        """Init."""

        self.criterion_func = criterion_func
        self.criterion_kwargs = criterion_kwargs

        self.criterion_grad = criterion_grad
        self.criterion_grad_kwargs = criterion_grad_kwargs

        self.return_arm_index = return_arm_index

        if arms is not None:

            self.d = arms[0].shape[0]
            self._arms = arms
            self.K = len(arms)

            self._arm_entries = arms_to_arm_entries(self._arms)

        elif arm_entries is not None:

            self.d = len(arm_entries)
            self._arm_entries = arm_entries
            log10_K = np.sum([np.log10(len(entry_vals))
                              for entry_vals in arm_entries.values()])
            self.K = int(10**log10_K) if log10_K <= np.log10(MAX_K) else np.inf

            if self.K != np.inf:
                self._arms = arm_entries_to_arms(self._arm_entries)

            else:
                self._arms = None  # will break if a for loop on arms occurs
                self.return_arm_index = False
                warnings.warn(f"The required number of arms (K={self.K}) "
                              f"exceed the maximum authorized {MAX_K}.")

        else:
            raise ValueError("To init 'Arms' class, either pass 'arms'"
                             " and 'arm_entries', none of them was given.")

    def _select_arm(self, theta, func, criterion_kwargs=None, grad_func=None,
                    criterion_grad_kwargs=None):
        """Select an arm for the given criterion."""
        if self.return_arm_index:
            uu = [func(x_k, theta, **criterion_kwargs)
                  for x_k in self._arms]

            return np.argmin(uu)

        else:
            def f(x):
                return func(x, theta, **criterion_kwargs)

            def grad(x):
                return grad_func(x, theta, **criterion_grad_kwargs)

            x0 = np.zeros(self.d)
            bounds = [(np.min(entry_vals), np.max(entry_vals))
                      for entry_vals in self._arm_entries.values()]

            res = optimize.minimize(fun=f, jac=grad, x0=x0, method='L-BFGS-B',
                                    bounds=bounds)

            return proj_on_arm_entries(res.x, self._arm_entries)

    def select_default_arm(self):
        """Return the selected arm with the lowest value for each coordinate
        (the 0th arm by convention)."""
        if self.return_arm_index:
            return 0

        else:
            arm_to_return = []
            for entry_values in self._arm_entries.values():
                arm_to_return.append(np.min(entry_values))

            return np.array(arm_to_return)

    def select_arm(self, theta, **kwargs):
        """Return the selected arm or its index from the given criterion."""

        criterion_kwargs = kwargs.get('criterion_kwargs',
                                      self.criterion_kwargs)
        criterion_grad_kwargs = kwargs.get('criterion_kwargs',
                                           self.criterion_grad_kwargs)

        selected_k_or_arm = self._select_arm(
                                theta=theta, func=self.criterion_func,
                                criterion_kwargs=criterion_kwargs,
                                grad_func=self.criterion_grad,
                                criterion_grad_kwargs=criterion_grad_kwargs)

        return selected_k_or_arm

    def best_arm(self, theta, **kwargs):
        """Return the estimated best arm or its index."""

        criterion_kwargs = kwargs.get('criterion_kwargs',
                                      self.criterion_kwargs)
        criterion_grad_kwargs = kwargs.get('criterion_kwargs',
                                           self.criterion_grad_kwargs)

        selected_k_or_arm = self._select_arm(
                                theta=theta, func=f_neg_scalar_prod,
                                criterion_kwargs=criterion_kwargs,
                                grad_func=grad_neg_scalar_prod,
                                criterion_grad_kwargs=criterion_grad_kwargs)

        return selected_k_or_arm
