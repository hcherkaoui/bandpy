""" Define the arm class and routines. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from scipy import optimize

from ._criterions import f_neg_scalar_prod, grad_neg_scalar_prod
from .utils import proj_on_arm_entries


DEFAULT_ARM_IDX = 0
MAX_K = 10000


def _select_arm(return_arm_index, arms, arm_entries, theta, func,
                criterion_kwargs, grad_func=None, criterion_grad_kwargs=None):
    """Select an arm for the given criterion."""
    if return_arm_index:
        uu = [func(x_k, theta, **criterion_kwargs) for x_k in arms]
        return np.argmin(uu)

    else:
        def f(x):
            return func(x, theta, **criterion_kwargs)

        if (grad_func is not None) and (criterion_grad_kwargs is not None):
            def grad(x):
                return grad_func(x, theta, **criterion_grad_kwargs)

        else:
            grad = None

        x0 = np.zeros_like(theta)
        bounds = [(np.min(entry_vals), np.max(entry_vals))
                  for entry_vals in arm_entries.values()]

        res = optimize.minimize(fun=f, jac=grad, x0=x0, method='L-BFGS-B',
                                bounds=bounds)

        return proj_on_arm_entries(res.x, arm_entries)


def quadratic_arm_to_arm(arm):
    """Convert an quadratic arm to a linear one."""
    return arm[int((len(arm) - 1) / 2):-1]


def arm_to_quadratic_arm(arm):
    """Convert arm to a quadratic one."""
    d = len(arm)

    q_arm = np.empty(shape=(2 * d + 1, 1), dtype=float)

    q_arm[:d, 0] = arm.ravel()**2
    q_arm[d:-1, 0] = arm.ravel()
    q_arm[-1, 0] = 1.0

    return q_arm


def arms_to_quadratic_arms(arms):
    """Convert arms to quadratic ones."""
    return [arm_to_quadratic_arm(arm) for arm in arms]


def arm_entries_to_quadratic_arm_entries(arm_entries):
    """Convert arm_entries to quadratic ones."""
    d = len(list(arm_entries.keys()))

    q_arm_entries = dict()
    for i in range(d):
        p_vals = arm_entries[f'P_{i}']

        q_arm_entries[f'P_{i}'] = np.copy(p_vals)
        q_arm_entries[f'P_{i}_2'] = np.square(p_vals)

    q_arm_entries['c'] = [1.0]

    return q_arm_entries


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
            raise ValueError("To init 'LinearArms' class, either pass 'arms'"
                             " and 'arm_entries', none of them was given.")

    def select_default_arm(self):
        """Return the selected arm with the lowest value for each coordinate
        (the 0th arm by convention)."""
        if self._arm_entries is None:
            return DEFAULT_ARM_IDX

        else:
            arm_to_return = []
            for entry_values in self._arm_entries.values():
                arm_to_return.append(np.min(entry_values))

            return np.array(arm_to_return)

    def select_arm(self, theta, **kwargs):
        """Return the selected arm or its index from the given criterion."""

        criterion_kwargs = kwargs.get('criterion_kwargs',
                                      self.criterion_kwargs)
        criterion_grad_kwargs = kwargs.get('criterion_kwargs',  # same kwargs for UCB  # noqa
                                           self.criterion_grad_kwargs)

        selected_k_or_arm = _select_arm(
                                return_arm_index=self.return_arm_index,
                                arms=self._arms,
                                arm_entries=self._arm_entries,
                                theta=theta, func=self.criterion_func,
                                criterion_kwargs=criterion_kwargs,
                                grad_func=self.criterion_grad,
                                criterion_grad_kwargs=criterion_grad_kwargs)

        return selected_k_or_arm

    def best_arm(self, theta, **kwargs):
        """Return the estimated best arm or its index."""

        criterion_kwargs = kwargs.get('criterion_kwargs',
                                      self.criterion_kwargs)
        criterion_grad_kwargs = kwargs.get('criterion_kwargs',  # same kwargs for UCB  # noqa
                                           self.criterion_grad_kwargs)

        selected_k_or_arm = _select_arm(
                                return_arm_index=self.return_arm_index,
                                arms=self._arms,
                                arm_entries=self._arm_entries,
                                theta=theta, func=f_neg_scalar_prod,
                                criterion_kwargs=criterion_kwargs,
                                grad_func=grad_neg_scalar_prod,
                                criterion_grad_kwargs=criterion_grad_kwargs)

        return selected_k_or_arm


class QuadraticArms(LinearArms):
    """Quadratic arms class."""

    def __init__(self, criterion_func, criterion_kwargs,
                 criterion_grad=None, criterion_grad_kwargs=None,
                 arms=None, arm_entries=None):
        """Init."""

        if arms is not None:
            self._linear_arms = arms
            self._linear_arm_entries = None
            arms = arms_to_quadratic_arms(arms)
            arm_entries = None

        elif arm_entries is not None:
            self._linear_arms = None
            self._linear_arm_entries = arm_entries
            arms = None
            arm_entries = arm_entries_to_quadratic_arm_entries(arm_entries)  # noqa

        else:
            raise ValueError("To init 'QuadraticArms' class, either pass "
                             "'arms' and 'arm_entries', none of them was "
                             "given.")

        super().__init__(criterion_func=criterion_func,
                         criterion_kwargs=criterion_kwargs,
                         criterion_grad=criterion_grad,
                         criterion_grad_kwargs=criterion_grad_kwargs,
                         arms=arms, arm_entries=arm_entries)

    def select_default_arm(self):
        """Return the selected arm with the lowest value for each coordinate
        (the 0th arm by convention)."""
        if self._linear_arm_entries is None:
            return DEFAULT_ARM_IDX

        else:
            arm_to_return = []
            for entry_values in self._linear_arm_entries.values():
                arm_to_return.append(np.min(entry_values))

            return np.array(arm_to_return)

    def select_arm(self, theta, **kwargs):
        """Return the selected arm or its index from the given criterion."""

        criterion_kwargs = kwargs.get('criterion_kwargs',
                                      self.criterion_kwargs)
        criterion_grad_kwargs = kwargs.get('criterion_kwargs',  # same kwargs for UCB  # noqa
                                           self.criterion_grad_kwargs)

        selected_k_or_quadratic_arm = _select_arm(
                                return_arm_index=self.return_arm_index,
                                arms=self._arms,
                                arm_entries=self._arm_entries,
                                theta=theta, func=self.criterion_func,
                                criterion_kwargs=criterion_kwargs,
                                grad_func=self.criterion_grad,
                                criterion_grad_kwargs=criterion_grad_kwargs)

        if isinstance(selected_k_or_quadratic_arm, np.ndarray):

            selected_k_or_arm = quadratic_arm_to_arm(selected_k_or_quadratic_arm)  # noqa
        else:
            selected_k_or_arm = selected_k_or_quadratic_arm

        return selected_k_or_arm

    def best_arm(self, theta, **kwargs):
        """Return the estimated best arm or its index."""

        criterion_kwargs = kwargs.get('criterion_kwargs',
                                      self.criterion_kwargs)
        criterion_grad_kwargs = kwargs.get('criterion_kwargs',  # same kwargs for UCB  # noqa
                                           self.criterion_grad_kwargs)

        selected_k_or_arm = _select_arm(
                                return_arm_index=self.return_arm_index,
                                arms=self._arms,
                                arm_entries=self._arm_entries,
                                theta=theta, func=f_neg_scalar_prod,
                                criterion_kwargs=criterion_kwargs,
                                grad_func=grad_neg_scalar_prod,
                                criterion_grad_kwargs=criterion_grad_kwargs)

        return selected_k_or_arm
