""" Define all check utility functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np


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
