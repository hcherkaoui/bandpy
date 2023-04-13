"""Testing module for the utils functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import re
import pytest
import numpy as np

from bandpy.utils import (convert_grid_to_list, _generate_list_of_nd_array,
                          generate_gaussian_arm_entries, arms_to_arm_entries,
                          arm_entries_to_arms, proj_on_arm_entries,
                          pytest_set_up)


def test_arms_and_arm_entries_converter():
    """ Test the convert function of arms <-> arm_entries. """
    ref_arm_entries = {'p_0': np.array([1, 2, 3]),
                       'p_1': np.array([0, 1]),
                       'p_2': np.array([1.1, 1.2, 1.3]),
                       }

    arms = arm_entries_to_arms(ref_arm_entries)
    arm_entries = arms_to_arm_entries(arms)

    for p_name in arm_entries.keys():
        np.testing.assert_allclose(ref_arm_entries[p_name],
                                   arm_entries[p_name], rtol=1e-5, atol=1e-3)


def test_convert_grid_to_list():
    """Test convert_grid_to_list."""
    grid = {'a': [0, 1],
            'b': [2.1, 2.2],
            }

    list_of_dict_ref = [{'a': 0, 'b': 2.1},
                        {'a': 0, 'b': 2.2},
                        {'a': 1, 'b': 2.1},
                        {'a': 1, 'b': 2.2},
                        ]

    assert list_of_dict_ref == convert_grid_to_list(grid)


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test__generate_list_of_nd_array(d, seed):
    """Test generate_gaussian_arms."""
    K = 5
    list_arrays = _generate_list_of_nd_array(K, d, seed=seed)

    assert isinstance(list_arrays, list)
    assert len(list_arrays) == K
    for arm in list_arrays:
        assert (d, 1) == arm.shape


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_generate_gaussian_arm_entries(d, seed):
    n_vals_per_dim = 3
    arm_entries = generate_gaussian_arm_entries(n_vals_per_dim=n_vals_per_dim,
                                                d=d, seed=seed)

    r = re.compile('p_*')

    assert isinstance(arm_entries, dict)
    assert all([isinstance(v, np.ndarray) for k, v in arm_entries.items()])
    assert all([r.match(k) is not None for k, v in arm_entries.items()])


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_proj_on_arm_entries(d, seed):
    """Test 'proj_on_arm_entries'."""
    set_up = pytest_set_up(d=d, seed=seed)

    x_k = set_up['x_k']

    n_vals_per_dim = 3
    arm_entries = generate_gaussian_arm_entries(n_vals_per_dim=n_vals_per_dim,
                                                d=d, seed=seed)
    proj_x_k = proj_on_arm_entries(x_k, arm_entries)

    arms = arm_entries_to_arms(arm_entries)
    k = np.argmin([np.sum(np.abs(x_k - arm)) for arm in arms])
    proj_x_k_ref = arms[k]

    np.testing.assert_array_equal(proj_x_k, proj_x_k_ref)
