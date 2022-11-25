"""Testing module for the criterion functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest
import numpy as np
from scipy.optimize import approx_fprime

from bandpy.utils import tests_set_up
from bandpy.criterions import (f_neg_ucb, grad_neg_ucb, f_neg_scalar_prod,
                               grad_neg_scalar_prod)


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_f_neg_scalar_prod(d, seed):
    """ Test the function of 'f_neg_scalar_prod'. """
    set_up = tests_set_up(d=d, seed=seed)

    inv_A = set_up['inv_A']
    x_k = set_up['x_k']
    theta = set_up['theta']

    kwargs = dict(alpha=0.0, inv_A=inv_A)

    ref_scalar_prod_ = f_neg_ucb(x_k, theta, **kwargs)
    scalar_prod_ = f_neg_scalar_prod(x_k, theta, **kwargs)

    np.testing.assert_allclose(ref_scalar_prod_, scalar_prod_, rtol=1e-5,
                               atol=1e-3)


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_f_neg_ucb(d, seed):
    """ Test the function of 'f_neg_ucb'. """
    set_up = tests_set_up(d=d, seed=seed)

    inv_A = set_up['inv_A']
    x_k = set_up['x_k']
    theta = set_up['theta']
    alpha = set_up['alpha']

    kwargs = dict(alpha=alpha, inv_A=inv_A)

    x_k = x_k.ravel()
    theta = theta.ravel()

    f_ucb_x_ref = theta.T.dot(x_k) + alpha * np.sqrt(x_k.T.dot(inv_A).dot(x_k))
    f_ucb_x = - f_neg_ucb(x_k, theta, **kwargs)

    np.testing.assert_allclose(f_ucb_x, f_ucb_x_ref, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_grad_neg_ucb(d, seed):
    """ Test the gradient of 'grad_neg_ucb'. """
    set_up = tests_set_up(d=d, seed=seed)

    inv_A = set_up['inv_A']
    x_k = set_up['x_k']
    theta = set_up['theta']
    alpha = set_up['alpha']

    kwargs = dict(alpha=alpha, inv_A=inv_A)

    def finite_grad(x):
        def f(x):
            x = x.reshape((d, 1))
            return f_neg_ucb(x, theta, **kwargs)
        grad_ = approx_fprime(xk=x.ravel(), f=f, epsilon=1.0e-6)
        return grad_.reshape((d, 1))

    finite_grad_x_k = finite_grad(x_k)
    grad_x_k = grad_neg_ucb(x_k, theta, **kwargs)

    np.testing.assert_allclose(finite_grad_x_k, grad_x_k, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_grad_neg_scalar_prod(d, seed):
    """ Test the gradient of 'grad_neg_scalar_prod'. """
    set_up = tests_set_up(d=d, seed=seed)

    inv_A = set_up['inv_A']
    x_k = set_up['x_k']
    theta = set_up['theta']
    alpha = set_up['alpha']

    kwargs = dict(alpha=alpha, inv_A=inv_A)

    def finite_grad(x):
        def f(x):
            x = x.reshape((d, 1))
            return f_neg_scalar_prod(x, theta, **kwargs)
        grad_ = approx_fprime(xk=x.ravel(), f=f, epsilon=1.0e-6)
        return grad_.reshape((d, 1))

    finite_grad_x_k = finite_grad(x_k)
    grad_x_k = grad_neg_scalar_prod(x_k, theta, **kwargs)

    np.testing.assert_allclose(finite_grad_x_k, grad_x_k, rtol=1e-5, atol=1e-3)
