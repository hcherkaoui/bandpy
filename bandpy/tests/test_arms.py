"""Testing module for the utility functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest
import numpy as np

from bandpy.arms import LinearArms
from bandpy.criterions import f_neg_ucb, grad_neg_ucb
from bandpy.utils import (tests_set_up, generate_gaussian_arm_entries,
                          arm_entries_to_arms)


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_LinearArms(d, seed):
    """ Test the check random state. """
    # set set-up
    set_up = tests_set_up(d=d, seed=seed)

    theta = set_up['theta']
    inv_A = set_up['inv_A']
    alpha = set_up['alpha']
    n_vals_per_dim = 2

    criterion_kwargs = dict(alpha=alpha, inv_A=inv_A)
    criterion_grad_kwargs = dict(alpha=alpha, inv_A=inv_A)

    # define arm_entries and corresponding arms
    arm_entries = generate_gaussian_arm_entries(n_vals_per_dim=n_vals_per_dim,
                                                d=d, seed=seed)
    arms = arm_entries_to_arms(arm_entries)
    K = len(arms)

    params_base = dict(criterion_func=f_neg_ucb,
                       criterion_kwargs=criterion_kwargs,
                       criterion_grad=grad_neg_ucb,
                       criterion_grad_kwargs=criterion_grad_kwargs)

    all_linear_arms = [LinearArms(**dict(**params_base,
                                         **dict(arms=arms,
                                                arm_entries=None))),
                       LinearArms(**dict(**params_base,
                                         **dict(arms=None,
                                                arm_entries=arm_entries))),
                       ]

    # define the default arm
    default_k = 0
    default_arm = np.array([np.min(val_entries)
                            for val_entries in arm_entries.values()])

    # manually find the best arm (for the given 'theta')
    best_k = np.argmax([theta.T.dot(arm) for arm in arms])
    best_arm = arms[best_k]

    # manually find the 'selected' arm (for the given 'inv_A' and 'theta')
    uu = []
    for arm in arms:
        _f_ucb_arm = theta.T.dot(arm)
        _f_ucb_arm += alpha * np.sqrt(arm.T.dot(inv_A).dot(arm))
        uu.append(_f_ucb_arm)

    selected_k = np.argmax(uu)
    selected_arm = arms[selected_k]

    # test all the 'linear_arms'
    for linear_arms in all_linear_arms:

        # check identical arm_entries or identical arms
        if linear_arms.return_arm_index:
            for arm, arm_ref in zip(linear_arms._arms, arms):
                np.testing.assert_array_equal(arm, arm_ref)

        else:
            assert linear_arms._arm_entries.keys() == arm_entries.keys()

            for vals, vals_ref in zip(linear_arms._arm_entries.values(),
                                      arm_entries.values()):
                np.testing.assert_array_equal(vals, vals_ref)

        # check identical K
        assert linear_arms.K == K

        # check identical d
        assert linear_arms.d == d

        # check agreement on the best arm (for a given theta)
        if linear_arms.return_arm_index:
            assert linear_arms.best_arm(theta) == best_k

        else:
            np.testing.assert_array_equal(linear_arms.best_arm(theta),
                                          best_arm)

        # check agreement on the default arm
        if linear_arms.return_arm_index:
            assert linear_arms.select_default_arm() == default_k

        else:
            np.testing.assert_array_equal(linear_arms.select_default_arm(),
                                          default_arm)

        # check agreement on the selected arm (for a given theta and inv_A)
        if linear_arms.return_arm_index:
            assert linear_arms.select_arm(theta) == selected_k

        else:
            np.testing.assert_array_equal(linear_arms.select_arm(theta),
                                          selected_arm)
