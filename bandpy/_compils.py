""" Define all pre-compiled functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
import numba


@numba.jit((numba.float64[:, :],
            numba.float64[:, :]), nopython=True, cache=True, fastmath=True)
def cholesky_rank_one_update(L, x):
    """Update of the Cholesky decomposition of A to match the one of
    A + x @ x^T."""
    L = np.copy(L.T)
    x = np.copy(x.ravel())

    len_x = len(x)
    for i in range(len_x):

        r = np.sqrt(L[i, i]**2 + x[i]**2)
        c = r / L[i, i]
        s = x[i] / L[i, i]

        L[i, i] = r
        L[i, i + 1:] = (L[i, i + 1:] + s * x[i + 1:]) / c
        x[i + 1:] = c * x[i + 1:] - s * L[i, i + 1:]

    return L.T


@numba.jit((numba.float64[:, :],
            numba.float64,
            numba.float64[:, :]), nopython=True, cache=True, fastmath=True)
def det_rank_one_update(inv_A, det_A, x):
    """Determinant update for the matrix ."""
    x = x.ravel()
    return det_A * (1.0 + x.T.dot(inv_A).dot(x))


@numba.jit((numba.float64[:, :],
            numba.float64[:, :]), nopython=True, cache=True, fastmath=True)
def sherman_morrison(inv_A, x):
    """Sherman-Morrison identity to compute the inverse of A + xxT."""
    inv_A_x = inv_A.dot(x)
    return inv_A - inv_A_x.dot(x.T.dot(inv_A)) / (1.0 + x.T.dot(inv_A_x))


@numba.jit((numba.float64[:, :],
            numba.float64[:, :]), nopython=True, cache=True, fastmath=True)
def geigh(inv_A_i, cho_A_j):
    """Generalized eigen-values decomposition for symetric postive-definite
    matrices."""
    lambdas, phi = np.linalg.eigh(cho_A_j.T.dot(inv_A_i).dot(cho_A_j))
    return lambdas, cho_A_j.dot(phi)


@numba.jit((numba.float64,
            numba.float64[:],
            numba.float64[:]), nopython=True, cache=True, fastmath=True)
def K_f(s, lambdas, v):
    """K function."""
    return 1.0 - np.sum(v * v * s * (1.0 - s) / (1.0 + s * (lambdas - 1.0)))


@numba.jit((numba.float64,
            numba.float64[:],
            numba.float64[:]), nopython=True, cache=True, fastmath=True)
def K_f_prime(s, lambdas, v):
    """K function first derivative."""
    return - np.sum(v * v * (-(lambdas - 1.0) * s**2 - 2.0 * s + 1) / (1.0 + (lambdas - 1.0) * s)**2)  # noqa


@numba.jit((numba.float64,
            numba.float64[:],
            numba.float64[:]), nopython=True, cache=True, fastmath=True)
def K_f_second(s, lambdas, v):
    """K function second derivative."""
    return - np.sum(v * v * (- 2.0 * (lambdas - 1.0) - 2.0) / (1.0 + (lambdas - 1.0) * s)**3)  # noqa


@numba.jit((numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :]), nopython=True, cache=True, fastmath=True)
def K_min(inv_A_i, cho_A_j, theta_i, theta_j):
    """Minimization of the function K_f."""
    lambdas, phi = geigh(inv_A_i, cho_A_j)
    theta_ij = (theta_i - theta_j).ravel()
    v = phi.T.dot(theta_ij)

    lambdas = lambdas.ravel()
    v = v.ravel()

    s, maxiter, eps = 0.5, 10, 1e-6
    for _ in range(maxiter):

        K_f_prime_s = K_f_prime(s, lambdas, v)

        if np.abs(K_f_prime_s) < eps:
            break

        s -= K_f_prime_s / K_f_second(s, lambdas, v)

        s = 0.0 if s < 0.0 else s
        s = 1.0 if s > 1.0 else s

    return s, K_f(s, lambdas, v)
