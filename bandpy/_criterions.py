""" Define bandit arm selection criterion functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
import numba


def f_neg_scalar_prod(x, theta, **kwargs):
    """Theta scalar product function criterion."""
    x = x.ravel()
    theta = theta.ravel()

    return - theta.T.dot(x)


def grad_neg_scalar_prod(x, theta, **kwargs):
    """Gradient of the theta scalar product function criterion."""
    return - theta


@numba.jit(numba.float64(numba.float64,
                         numba.float64[:],
                         numba.float64[:],
                         numba.float64[:, :]),
           nopython=True, cache=True, fastmath=True)
def _f_ucb(alpha, x, theta, inv_A):
    _f = theta.T.dot(x) + alpha * np.sqrt(x.T.dot(inv_A).dot(x))
    return numba.float64(_f)


def f_neg_ucb(x, theta, **kwargs):
    """UCB function criterion."""
    alpha = kwargs['alpha']
    inv_A = kwargs['inv_A']
    return - _f_ucb(alpha, x.ravel(), theta.ravel(), inv_A)


@numba.jit(numba.float64[:, :](numba.float64,
                               numba.float64[:],
                               numba.float64[:],
                               numba.float64[:, :]),
           nopython=True, cache=True, fastmath=True)
def _grad_ucb(alpha, x, theta, inv_A):
    d, _ = inv_A.shape
    _grad = theta + alpha * inv_A.dot(x) / np.sqrt(x.T.dot(inv_A).dot(x))
    return _grad.reshape((d, 1))


def grad_neg_ucb(x, theta, **kwargs):
    """Gradient of the UCB function criterion."""
    alpha = kwargs['alpha']
    inv_A = kwargs['inv_A']
    return - _grad_ucb(alpha, x.ravel(), theta.ravel(), inv_A)
