"""Testing module for the utility functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest
import numpy as np

from bandpy._checks import check_random_state


@pytest.mark.parametrize("seed", [0, 1])
def test_check_random_state(seed):
    """Test the check random state."""
    rng = check_random_state(seed)
    assert isinstance(rng, np.random.Generator)
    rng = check_random_state(np.random)
    assert isinstance(rng, np.random.Generator)
    rng = check_random_state(3)
    assert isinstance(rng, np.random.Generator)
    rng = check_random_state(check_random_state(seed))
    assert isinstance(rng, np.random.Generator)
    with pytest.raises(ValueError):
        check_random_state("foo")
