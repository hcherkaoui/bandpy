""" Define all pre-compiled functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
import numba


@numba.jit((numba.float64[:, :],
            numba.float64[:, :]), nopython=True, cache=True, fastmath=True)
def _fast_inv_sherman_morrison(inv_A, x):  # pragma: no cover
    """Sherman-Morrison identity to compute the inverse of A + xxT."""
    inv_A_x = inv_A.dot(x)
    return inv_A - inv_A_x.dot(x.T.dot(inv_A)) / (1.0 + x.T.dot(inv_A_x))


@numba.jit((numba.float64,
            numba.float64[:, :],
            numba.float64[:, :]), nopython=True, cache=True, fastmath=True)
def _K_func(s, v_squared, lambdas):
    norm_ = np.sum(v_squared * s * (1.0 - s) / (1.0 + s * (lambdas - 1.0)))
    return 1.0 - norm_
