""" Define all misc utility functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import cProfile
import itertools
import numpy as np
import pandas as pd
from joblib import Memory
from matrix_factorization import KernelMF

from ._checks import check_random_state


def _tolerant_concat(arrs):
    """Concatenate time-serie of different lengths."""
    n_arrs = len(arrs)
    all_lengths = [len(arr) for arr in arrs]
    concat_arr = np.ma.empty((np.max(all_lengths), n_arrs))
    concat_arr.mask = True

    for i, arr in enumerate(arrs):
        concat_arr[: len(arr), i] = arr

    return concat_arr


def tolerant_median(arrs):
    """Compute the mean of time-serie of different length."""
    return np.ma.median(_tolerant_concat(arrs), axis=-1)


def tolerant_mean(arrs):
    """Compute the mean of time-serie of different length."""
    return _tolerant_concat(arrs).mean(axis=-1)


def tolerant_std(arrs):
    """Compute the std of time-serie of different length."""
    return _tolerant_concat(arrs).std(axis=-1)


def tolerant_min(arrs):
    """Compute the min of time-serie of different length."""
    return _tolerant_concat(arrs).min(axis=-1)


def tolerant_max(arrs):
    """Compute the max of time-serie of different length."""
    return _tolerant_concat(arrs).max(axis=-1)


def tolerant_stats(arrs):
    """Compute the mean, std, max and min of time-serie of different length."""
    all_lengths = [len(arr) for arr in arrs]

    concat_arr = _tolerant_concat(arrs)

    results = [
        concat_arr.mean(axis=-1),
        concat_arr.std(axis=-1),
        concat_arr.max(axis=-1),
        concat_arr.min(axis=-1),
        np.array(all_lengths),
    ]

    return results


def generate_thetas_arms_1(K, d, n_thetas=2, offset=0.0, seed=None):
    """Generate a set of thetas and arms."""
    rng = check_random_state(seed)

    thetas = [rng.normal(size=(d, 1)) + offset for _ in range(n_thetas)]
    arms = [rng.normal(size=(d, 1)) for _ in range(K)]

    return thetas, arms


def generate_thetas_arms_2(
    K, d, angle=np.pi, tiny_angle=np.pi / 16.0, seed=None
):  # noqa
    """Generate a set of thetas and arms."""
    rng = check_random_state(seed)

    thetas = [
        np.array([1.0] + [0.0] * (d - 1)).reshape((d, 1)),
        np.array([np.cos(angle), np.sin(angle)] + [0.0] * (d - 2)).reshape((d, 1)),
    ]

    arms = [np.array([1.0, 0.0, np.sin(tiny_angle)] + list(rng.normal(size=(d - 3))))]
    arms += [
        np.array(
            [np.cos(angle), np.sin(angle), np.sin(tiny_angle)] + list(rng.normal(size=(d - 3)))
        )
    ]
    arms += [0.5 * (arms[-2] + arms[-1])]
    arms += [-arms[-1]]
    arms += [np.array([0.0, 0.0, 0.0] + list(rng.normal(size=(d - 3)))) for _ in range(K - 4)]

    return thetas, arms


def generate_thetas_arms_3(
    K, d, n_thetas=2, angle=np.pi, coef=0.5, offset=0.0, seed=None
):
    """Generate arms and thetas."""
    rng = check_random_state(seed)

    msg = (
        "Dimension 'd' should be greater than the resquest number of thetas"
        " 'n_thetas'"
    )
    assert n_thetas <= d, msg

    thetas = []
    for i in range(1, n_thetas + 1):
        if i % 2 != 0:
            theta = np.zeros(shape=(d, 1), dtype=float)
            theta[i - 1] = 1.0

        else:
            theta = np.zeros(shape=(d, 1), dtype=float)
            theta[i - 2] = np.cos(angle)
            theta[i - 1] = np.sin(angle)

        thetas.append(theta + offset)

    arms = []
    for theta in thetas:
        norm_theta = np.linalg.norm(theta)
        arms.extend(
            [
                theta + coef * norm_theta * rng.normal(size=theta.shape)
                for _ in range(int(K / n_thetas))
            ]
        )

    return thetas, arms


def _fill_mising_values(data, K, col_name=None):
    """Fill missing values with matrix factorization approach."""

    if col_name is not None:
        data = data.rename(columns=col_name)

    def get_not_observed_arms(user_data):
        item_id = list(set(range(K)) - set(np.unique(user_data["item_id"])))
        user_id = [int(user_data["user_id"].iloc[0])] * len(item_id)
        df = pd.DataFrame(np.c_[user_id, item_id], columns=["user_id", "item_id"])
        return df.reset_index()

    X_train = data[["user_id", "item_id"]]
    y_train = data["rating"]

    filler = KernelMF(n_epochs=20, n_factors=100, verbose=0, lr=0.001, reg=0.005)
    filler = filler.fit(X_train, y_train)

    results = data.groupby("user_id").apply(get_not_observed_arms)
    ratings = filler.predict(results)
    user_id = results.loc[:, "user_id"].to_numpy()
    item_id = results.loc[:, "item_id"].to_numpy()

    data_ = pd.DataFrame(
        np.c_[ratings, user_id, item_id], columns=["rating", "user_id", "item_id"]
    )
    data_.loc[:, "user_id"] = data_.loc[:, "user_id"].astype(int)
    data_.loc[:, "item_id"] = data_.loc[:, "item_id"].astype(int)
    data_.loc[:, "rating"] = np.round(data_.loc[:, "rating"], 0)

    return pd.concat([data, data_], ignore_index=True)


fill_mising_values = Memory("__cache__", verbose=0).cache(_fill_mising_values)


def format_duration(seconds):
    """Converts a duration in seconds to HH:MM:SS format."""
    s, ns = divmod(seconds, 1)
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s {int(1000.0 * ns):03d}ms"


def profile_me(func):  # pragma: no cover
    """Profiling decorator, produce a report <func-name>.profile to be open as
    Place @profile_me on top of the desired function, then:
    'python -m snakeviz <func-name>.profile'
    Parameters
    ----------
    func : func, function to profile
    """

    def profiled_func(*args, **kwargs):
        filename = func.__name__ + ".profile"
        prof = cProfile.Profile()
        ret = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(filename)
        return ret

    return profiled_func


def convert_grid_to_list(grid):
    """Convert a dict of list (grid) to a list of dict (combination)."""
    combs = itertools.product(*grid.values())
    return [dict(zip(grid.keys(), p)) for p in combs]


def get_d(arms=None, arm_entries=None):
    """Get d from 'arms' and 'arm_entries' in a safe way."""
    if arms is not None:
        return arms[0].shape[0]

    elif arm_entries is not None:
        return len(arm_entries)

    else:
        raise ValueError(
            "To init 'LinUCB' class, either pass 'arms'"
            " and 'arm_entries', none of them was given."
        )


def arms_to_arm_entries(arms):
    """Convert 'arms' to 'arm_entries'."""
    arm_entries = dict()
    for i, entry_vals in enumerate(zip(*arms)):
        arm_entries[f"p_{i}"] = np.sort(np.unique(entry_vals))
    return arm_entries


def arm_entries_to_arms(arm_entries):
    """Convert 'arm_entries' to 'arms'."""
    combs = itertools.product(*arm_entries.values())
    return [np.array(p).reshape((len(p), 1)) for p in combs]


def proj_on_arm_entries(x, arm_entries):
    """Project a vector on a discretize grid specified by 'arm_entries'."""
    proj_x = []
    for i, entry_vals in enumerate(arm_entries.values()):
        dist_ = np.abs(x[i] - entry_vals)
        proj_x.append(entry_vals[np.argmin(dist_)])

    return np.array(proj_x, dtype=float).reshape((len(proj_x), 1))


def _generate_list_of_nd_array(n, d, offset=0.0, seed=None):
    rng = check_random_state(seed)
    return [rng.normal(size=(d, 1)) + offset for _ in range(n)]


def generate_gaussian_arms(K, d, seed=None):  # pragma: no cover
    """Generate Gaussian 'arms'."""
    return _generate_list_of_nd_array(n=K, d=d, offset=0.0, seed=seed)


def generate_gaussian_thetas(n_thetas, d, theta_offset=0.0, seed=None):
    """Generate Gaussian 'thetas'."""
    return _generate_list_of_nd_array(n=n_thetas, d=d, offset=theta_offset, seed=seed)


def generate_gaussian_arm_entries(n_vals_per_dim, d, seed=None):
    """Generate Gaussian 'arm_entries'."""
    rng = check_random_state(seed)
    arm_entries = dict()
    for i in range(d):
        arm_entries[f"p_{i}"] = np.sort(rng.normal(size=(n_vals_per_dim, 1)))
    return arm_entries


def pytest_set_up(d=2, seed=None):
    """Synthetic set-up for the unittests."""
    rng = check_random_state(seed)
    set_up = dict()

    set_up["seed"] = seed
    set_up["rng"] = rng
    set_up["inv_A"] = np.eye(d)
    set_up["x_k"] = rng.normal(size=(d, 1))
    set_up["theta"] = rng.normal(size=(d, 1))
    set_up["alpha"] = 1.0
    set_up["t"] = 1
    set_up["lbda"] = 1.0

    return set_up
