"""Testing module for the runners functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest
import numpy as np

from bandpy import agents
from bandpy._arms import DEFAULT_ARM_IDX
from bandpy.utils import tests_set_up


@pytest.mark.parametrize('K', [2, 5, 10])
@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
def test_LinUniform(d, K, seed):
    """Test the LinUniform agent."""
    set_up = tests_set_up(d=d, seed=seed)

    seed = set_up['seed']
    rng = set_up['rng']
    lbda = set_up['lbda']
    te = set_up['te']

    arms = [rng.randn(d) for _ in range(K)]

    agent_ = agents.LinUniform(arms=arms, arm_entries=None, lbda=lbda, te=te,
                               seed=seed)

    assert DEFAULT_ARM_IDX == agent_.select_default_arm()

    observation = {'last_arm_pulled': 0, 't': 10}
    reward = 0.1
    k = agent_.act(observation, reward)

    assert k in np.arange(K)
