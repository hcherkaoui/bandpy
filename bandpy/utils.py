""" Define all check utility functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import cProfile
import itertools
import numpy as np
import pandas as pd
from joblib import Memory
from matrix_factorization import KernelMF


def check_N_and_agent_names(N, agent_names):
    """ Check N and 'agent_names'."""

    if (N is None) and (agent_names is None):
        raise ValueError("Number of agents N and 'agent_names' can't be "
                         "both set to None.")

    elif (N is not None) and (agent_names is None):
        return N, [f'agent_{i}' for i in range(N)]

    elif (N is None) and (agent_names is not None):
        return len(agent_names), agent_names

    else:
        if N != len(agent_names):
            raise ValueError(f"Number of agents N and number of agents in"
                             f" 'agent_names' should be equal,"
                             f" got {N} and {len(agent_names)}.")

        else:
            return N, agent_names


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance
    Return
    ------
    random_instance : random-instance used to initialize the analysis
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"{seed} cannot be used to seed a "
                     f"numpy.random.RandomState instance")


def check_actions(actions):
    """ Check if the 'actions' are properly formatted."""
    if not isinstance(actions, dict):
        raise ValueError(f"'actions' should be a dict, got {type(actions)}")

    for agent_name, action in actions.items():
        if not isinstance(agent_name, str):
            raise ValueError(f"Agent name should be str, got "
                             f"{type(agent_name)}")

        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action for agent '{agent_name}' should be type"
                             f" int, got {type(action)}")

    return actions


def tolerant_stats(arrs):
    """Compute the means and stds of time-serie of different length."""
    n_arrs = len(arrs)
    all_lengths = [len(arr) for arr in arrs]
    concat_arr = np.ma.empty((np.max(all_lengths), n_arrs))
    concat_arr.mask = True

    for i, arr in enumerate(arrs):
        concat_arr[:len(arr), i] = arr

    results = [concat_arr.mean(axis=-1),
               concat_arr.std(axis=-1),
               concat_arr.max(axis=-1),
               concat_arr.min(axis=-1),
               np.array(all_lengths),
               ]

    return results


def _fill_mising_values(data, K, col_name=None):
    """Fill missing values with matrix factorization approach."""

    if col_name is not None:
        data = data.rename(columns=col_name)

    def get_not_observed_arms(user_data):
        item_id = list(set(range(K)) - set(np.unique(user_data['item_id'])))
        user_id = [int(user_data['user_id'].iloc[0])] * len(item_id)
        df = pd.DataFrame(np.c_[user_id, item_id],
                          columns=['user_id', 'item_id'])
        return df.reset_index()

    X_train = data[['user_id', 'item_id']]
    y_train = data['rating']

    filler = KernelMF(n_epochs=20, n_factors=100, verbose=0, lr=0.001,
                      reg=0.005)
    filler = filler.fit(X_train, y_train)

    results = data.groupby("user_id").apply(get_not_observed_arms)
    ratings = filler.predict(results)
    user_id = results.loc[:, 'user_id'].to_numpy()
    item_id = results.loc[:, 'item_id'].to_numpy()

    data_ = pd.DataFrame(np.c_[ratings, user_id, item_id],
                         columns=['rating', 'user_id', 'item_id'])
    data_.loc[:, 'user_id'] = data_.loc[:, 'user_id'].astype(int)
    data_.loc[:, 'item_id'] = data_.loc[:, 'item_id'].astype(int)
    data_.loc[:, 'rating'] = np.round(data_.loc[:, 'rating'], 0)

    return pd.concat([data, data_], ignore_index=True)


fill_mising_values = Memory('__cache__', verbose=0).cache(_fill_mising_values)


def profile_me(func):  # pragma: no cover
    """ Profiling decorator, produce a report <func-name>.profile to be open as
    Place @profile_me on top of the desired function, then:
    `python -m snakeviz  <func-name>.profile`
    Parameters
    ----------
    func : func, function to profile
    """
    def profiled_func(*args, **kwargs):
        filename = func.__name__ + '.profile'
        prof = cProfile.Profile()
        ret = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(filename)
        return ret
    return profiled_func


def convert_grid_to_list(grid):
    """ Convert a dict of list (grid) to a list of dict (combination)."""
    combs = itertools.product(*grid.values())
    return [dict(zip(grid.keys(), p)) for p in combs]
