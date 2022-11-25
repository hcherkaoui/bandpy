""" Define the arm class and routines. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from scipy import optimize

from .criterions import f_neg_scalar_prod, grad_neg_scalar_prod
from .utils import proj_on_arm_entries
from .__init__ import MAX_K


def _select_default_arm(arm_entries=None):
    """Return the 'default arm' defined by convention."""
    if arm_entries is None:
        return 0

    else:
        arm_to_return = []
        for entry_values in arm_entries.values():
            arm_to_return.append(np.min(entry_values))

        return np.array(arm_to_return)


class LinearArms:
    """Linear arms class."""

    def __init__(self, criterion_func, criterion_kwargs,
                 criterion_grad=None, criterion_grad_kwargs=None,
                 arms=None, arm_entries=None):
        """Init."""

        self.criterion_func = criterion_func
        self.criterion_kwargs = criterion_kwargs

        self.criterion_grad = criterion_grad
        self.criterion_grad_kwargs = criterion_grad_kwargs

        # In priority: based the 'act' method on an arms for loop
        if arms is not None:

            self._arms = arms
            self._arm_entries = None

            self.return_arm_index = True

            self.K = len(arms)
            self.d = arms[0].shape[0]

        # if 'arms' is not avalaible fall back on arm_entries
        elif arm_entries is not None:

            self._arms = None
            self._arm_entries = arm_entries

            self.return_arm_index = False

            log10_K = np.sum([np.log10(len(entry_vals))
                              for entry_vals in arm_entries.values()])
            self.K = int(10**log10_K) if log10_K <= np.log10(MAX_K) else np.inf
            self.d = len(arm_entries)

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

            if (grad_func is not None) and (criterion_grad_kwargs is not None):
                def grad(x):
                    return grad_func(x, theta, **criterion_grad_kwargs)

            else:
                grad = None

            x0 = np.zeros(self.d)
            bounds = [(np.min(entry_vals), np.max(entry_vals))
                      for entry_vals in self._arm_entries.values()]

            res = optimize.minimize(fun=f, jac=grad, x0=x0, method='L-BFGS-B',
                                    bounds=bounds)

            return proj_on_arm_entries(res.x, self._arm_entries)

    def select_default_arm(self):
        """Return the selected arm with the lowest value for each coordinate
        (the 0th arm by convention)."""
        return _select_default_arm(arm_entries=self._arm_entries)

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
