""" Define all check utility functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import warnings
import re
import numpy as np


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

        if not isinstance(action, (int, np.integer, np.ndarray)):
            raise ValueError(f"Action for agent '{agent_name}' should be type"
                             f" int or vector-like, got {type(action)}")

    return actions


def check_arms(arms, msg='arms'):
    """Check if 'arms' is properly formatted."""
    valid_format = (isinstance(arms, list) and
                    all([isinstance(arm, np.ndarray) for arm in arms]))

    if not valid_format:
        raise ValueError(f"'{msg}' should be list of 1d-arrays")

    return arms


def check_thetas(thetas, msg='thetas'):
    """Check if 'thetas' is properly formatted."""
    return check_arms(thetas, msg=msg)


def check_arm_entries(arm_entries):
    """Check if 'arm_entries' is properly formatted."""
    r = re.compile('p_*')

    valid_format = (isinstance(arm_entries, dict) and
                    all([isinstance(v, np.ndarray) and
                        (r.match(k) is not None)
                        for k, v in arm_entries.items()]))

    if not valid_format:
        raise ValueError("'arm_entries' should be dict with key "
                         "formatted as 'p_*' and values being "
                         "1d-arrays")

    return arm_entries


def check_thetas_and_n_thetas(d=None, thetas=None, n_thetas=None,
                              theta_offset=0.0, seed=None):
    """Check the requested 'thetas setting' is properly formatted."""

    # -all- the possible settings are check (even with redundancie)

    if (n_thetas is not None) and (thetas is not None):
        thetas = check_thetas(thetas)
        n_thetas = len(thetas)
        warnings.warn("'thetas' and 'n_thetas' specified: ignoring "
                      "'n_thetas'.")

    elif (n_thetas is None) and (thetas is not None):
        thetas = check_thetas(thetas)
        n_thetas = len(thetas)

    elif (n_thetas is not None) and (thetas is None):
        rng = check_random_state(seed)
        thetas = [rng.randn(d, 1) + theta_offset for _ in range(n_thetas)]

    elif (n_thetas is None) and (thetas is None):
        raise ValueError("'n_thetas' and 'thetas' should not be both set "
                         "to None simultaneously.")

    return thetas, n_thetas


def check_K_arms_arm_entries(d, arms=None, arm_entries=None, K=None,
                             seed=None):
    """Check the requested 'arms setting' is properly formatted."""

    # -all- the possible settings are check (even with redundancie)

    if (arms is None) and (K is None) and (arm_entries is None):
        raise ValueError("'K', 'arms' and 'arm_entries' should not be "
                         "both set to None simultaneously.")

    if (arms is not None) and (K is not None) and (arm_entries is not None):
        warnings.warn("'K', 'arms' and 'arm_entries' all specified: "
                      "ignoring 'K' and 'arm_entries'.")

        arms = check_arms(arms)
        arm_entries = None
        K = len(arms)
        return_arm_index = True

        return return_arm_index, arms, arm_entries, K

    if (arms is not None) and (K is not None) and (arm_entries is None):
        warnings.warn("'arms' and 'K' specified: ignoring 'K'.")

        arms = check_arms(arms)
        arm_entries = None
        K = len(arms)
        return_arm_index = True

        return return_arm_index, arms, arm_entries, K

    if (arms is not None) and (K is None) and (arm_entries is not None):
        warnings.warn("'arms' and 'arm_entries' specified: ignoring "
                      "'arm_entries'.")

        arms = check_arms(arms)
        arm_entries = None
        K = len(arms)
        return_arm_index = True

        return return_arm_index, arms, arm_entries, K

    if (arms is None) and (K is not None) and (arm_entries is not None):
        warnings.warn("'K' and 'arm_entries' specified: ignoring 'K'.")

        arms = None
        arm_entries = check_arm_entries(arm_entries)
        K = np.inf
        return_arm_index = False

        return return_arm_index, arms, arm_entries, K

    if (arms is not None) and (K is None) and (arm_entries is None):

        arms = check_arms(arms)
        arm_entries = None
        K = len(arms)
        return_arm_index = True

        return return_arm_index, arms, arm_entries, K

    if (arms is None) and (K is not None) and (arm_entries is None):

        rng = check_random_state(seed)
        arms = [rng.randn(d, 1) for _ in range(K)]
        arm_entries = None
        K = K
        return_arm_index = True

        return return_arm_index, arms, arm_entries, K

    if (arms is None) and (K is None) and (arm_entries is not None):

        arms = None
        arm_entries = check_arm_entries(arm_entries)
        K = np.inf
        return_arm_index = False

        return return_arm_index, arms, arm_entries, K
