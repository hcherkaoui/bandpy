"""Testing module for the compils functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest
import numpy as np
from scipy import optimize

from bandpy._compils import (
    cholesky_rank_one_update,
    det_rank_one_update,
    sherman_morrison,
    K_f,
    geigh,
    K_min,
)
from bandpy._checks import check_random_state


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("d", [2, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_cholesky_rank_one_update(n, d, seed):
    """Test the _cholesky_rank_one_update"""
    rng = check_random_state(seed)

    m = 10
    lbda = 1.0
    X = rng.normal(size=(d, m))
    A = lbda * np.eye(d) + X @ X.T
    L_test = np.linalg.cholesky(A)

    for x in [rng.normal(size=(d, 1)) for _ in range(n)]:
        A += x @ x.T

        L_ref = np.linalg.cholesky(A)
        L_test = cholesky_rank_one_update(L_test, x)

        np.testing.assert_allclose(L_test, L_ref)


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("d", [2, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_sherman_morrison(n, d, seed):
    """Test the sherman_morrison"""
    rng = check_random_state(seed)

    lbda = 1.0
    A = lbda * np.eye(d)
    inv_A_test = np.linalg.inv(A)

    for x in [rng.normal(size=(d, 1)) for _ in range(n)]:
        A += x @ x.T

        inv_A_ref = np.linalg.inv(A)
        inv_A_test = sherman_morrison(inv_A_test, x)

        np.testing.assert_allclose(inv_A_test, inv_A_ref)


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("d", [2, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_det_rank_one_update(n, d, seed):
    """Test the _det_rank_one_update"""
    rng = check_random_state(seed)

    m = 10
    lbda = 1.0
    X = rng.normal(size=(d, m))
    A = lbda * np.eye(d) + X @ X.T
    det_A_test = np.linalg.det(A)

    for x in [rng.normal(size=(d, 1)) for _ in range(n)]:
        inv_A = np.linalg.inv(A)
        A += x @ x.T

        det_A_ref = np.linalg.det(A)
        det_A_test = det_rank_one_update(inv_A, det_A_test, x)

        np.testing.assert_allclose(det_A_test, det_A_ref)


@pytest.mark.parametrize("N", [2, 10])
@pytest.mark.parametrize("K", [2, 10])
@pytest.mark.parametrize("d", [2, 10])
@pytest.mark.parametrize("T", [5, 10])
@pytest.mark.parametrize("lbda", [1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_K_f(N, K, d, T, lbda, seed):
    """Test the _K_func"""

    def K_f_ref(s, A_i, A_j, theta_i, theta_j):
        theta_ij = (theta_i - theta_j).ravel()
        A = 1.0 / (1.0 - s) * np.linalg.inv(A_i) + 1.0 / s * np.linalg.inv(A_j)
        return 1.0 - theta_ij.T @ np.linalg.inv(A) @ theta_ij

    rng = check_random_state(seed)

    A_init = lbda * np.eye(d)
    X = [rng.normal(size=(d, 1)) for _ in range(K)]

    net = dict()
    for i in range(N):
        A = np.copy(A_init)
        inv_A = np.linalg.inv(A_init)
        cho_A = np.linalg.cholesky(A_init)
        theta = rng.normal(size=(d, 1))
        net[i] = dict(A=A, inv_A=inv_A, cho_A=cho_A, theta=theta)

    for _ in range(T):
        i = rng.integers(low=0, high=N)
        k = rng.choice(K)
        x = X[k]

        net[i]["A"] += x @ x.T
        net[i]["inv_A"] = sherman_morrison(net[i]["inv_A"], x)
        net[i]["cho_A"] = cholesky_rank_one_update(net[i]["cho_A"], x)

        for j in range(N):
            for s in np.linspace(0.0 + 1e-6, 1.0 - 1e-6, 10):
                K_val_ref = K_f_ref(
                    s, net[i]["A"], net[j]["A"], net[i]["theta"], net[j]["theta"]
                )

                lambdas, phi = geigh(net[i]["inv_A"], net[j]["cho_A"])
                theta_ij = (net[i]["theta"] - net[j]["theta"]).ravel()
                v = phi.T.dot(theta_ij)
                K_val = K_f(s, lambdas, v)

                np.testing.assert_allclose(K_val, K_val_ref)


@pytest.mark.parametrize("d", [2, 10])
@pytest.mark.parametrize("seed", [0, 1])
def test_sign_K_min(d, seed):
    """Test the K_min sign agreement"""
    rng = check_random_state(seed)

    inv_A_i = np.eye(d)
    cho_A_j = np.eye(d)

    theta = rng.normal(size=(d, 1))
    theta_i = theta
    theta_j = theta
    _, f_min = K_min(inv_A_i, cho_A_j, theta_i, theta_j)
    assert f_min == 1.0

    theta = np.array([1.0] + [0.0] * (d - 1))[:, None]
    theta_i = theta
    theta_j = 0.5 * theta
    _, f_min = K_min(inv_A_i, cho_A_j, theta_i, theta_j)
    assert f_min > 0.0

    theta = rng.normal(size=(d, 1))
    theta_i = theta
    theta_j = 10.0 * theta
    _, f_min = K_min(inv_A_i, cho_A_j, theta_i, theta_j)
    assert f_min < 0.0


@pytest.mark.parametrize("N", [2, 10])
@pytest.mark.parametrize("K", [2, 10])
@pytest.mark.parametrize("d", [2, 10])
@pytest.mark.parametrize("T", [5, 10])
@pytest.mark.parametrize("lbda", [1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_K_min(N, K, d, T, lbda, seed):
    """Test the K_min function"""

    def K_min_ref(A_i, A_j, theta_i, theta_j):
        def f(s):
            theta_ij = (theta_i - theta_j).ravel()
            A = 1.0 / (1.0 - s) * np.linalg.inv(A_i)
            A += 1.0 / s * np.linalg.inv(A_j)
            return 1.0 - theta_ij.T @ np.linalg.inv(A) @ theta_ij

        res = optimize.minimize_scalar(
            f, bracket=None, bounds=[0.0, 1.0], method="bounded"
        )

        return res.x, res.fun

    rng = check_random_state(seed)
    A_init = lbda * np.eye(d)
    X = [rng.normal(size=(d, 1)) for _ in range(K)]

    net = dict()
    for i in range(N):
        A = np.copy(A_init)
        inv_A = np.linalg.inv(A_init)
        cho_A = np.linalg.cholesky(A_init)
        theta = rng.normal(size=(d, 1))
        net[i] = dict(A=A, inv_A=inv_A, cho_A=cho_A, theta=theta)

    for t in range(T):
        i = rng.integers(low=0, high=N)
        k = rng.choice(K)
        x = X[k]

        net[i]["A"] += x @ x.T
        net[i]["inv_A"] = sherman_morrison(net[i]["inv_A"], x)
        net[i]["cho_A"] = cholesky_rank_one_update(net[i]["cho_A"], x)

        for j in range(N):
            A_i, A_j = net[i]["A"], net[j]["A"]
            inv_A_i, cho_A_j = net[i]["inv_A"], net[j]["cho_A"]
            theta_i, theta_j = net[i]["theta"], net[j]["theta"]

            _, K_min_value_ref = K_min_ref(A_i, A_j, theta_i, theta_j)
            _, K_min_value = K_min(inv_A_i, cho_A_j, theta_i, theta_j)

            np.testing.assert_allclose(K_min_value, K_min_value_ref)
