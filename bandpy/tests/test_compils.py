"""Testing module for the compils functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest
import numpy as np
from scipy import linalg
from bandpy.compils import _fast_inv_sherman_morrison, _K_func
from bandpy.utils import tests_set_up
from bandpy.checks import check_random_state


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_fast_inv_sherman_morrison(d, seed):
    """Test the _fast_inv_sherman_morrison"""
    set_up = tests_set_up(d=d, seed=seed)

    inv_A = set_up['inv_A']
    x_k = set_up['x_k']

    inv_A_updated_ref = np.linalg.inv(inv_A + x_k.dot(x_k.T))
    inv_A_updated = _fast_inv_sherman_morrison(inv_A, x_k)

    np.testing.assert_allclose(inv_A_updated_ref, inv_A_updated, rtol=1e-5,
                               atol=1e-3)


@pytest.mark.parametrize('s', [0.1, 0.5, 0.9])
@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test__K_func(s, d, seed):
    """Test the _K_func"""
    rng = check_random_state(seed)

    A, B = rng.randn(d, d), rng.randn(d, d)

    inv_V_i = A.dot(A.T)
    inv_V_j = B.dot(B.T)
    theta_i = rng.randn(d, 1)
    theta_j = rng.randn(d, 1)
    eps_i = 1e-2
    eps_j = 1e-3

    def K_func_ref(s, inv_V_i, inv_V_j, theta_i, theta_j, eps_i, eps_j):
        theta_ij = theta_i - theta_j
        A = eps_j / (1.0 - s) * inv_V_j
        A += eps_i / s * inv_V_i
        norm_ = (theta_ij).T.dot(np.linalg.pinv(A)).dot(theta_ij)
        return 1.0 - float(norm_)

    def K_func_test(s, inv_V_i, inv_V_j, theta_i, theta_j, eps_i, eps_j):
        """K function."""
        d, _ = theta_i.shape  # either theta_i or theta_j
        theta_ij = theta_i - theta_j
        lambdas, phi = linalg.eigh(inv_V_j * eps_j, b=inv_V_i * eps_i)
        lambdas = lambdas.reshape((d, 1))
        phi = phi.reshape((d, d))
        v_squared = phi.T.dot(theta_ij) ** 2
        return float(_K_func(s, v_squared, lambdas))

    K_ref = K_func_ref(s, inv_V_i, inv_V_j, theta_i, theta_j, eps_i, eps_j)
    K_test = K_func_test(s, inv_V_i, inv_V_j, theta_i, theta_j, eps_i, eps_j)

    np.testing.assert_allclose(K_ref, K_test, rtol=1e-5, atol=1e-3)
