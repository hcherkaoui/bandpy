"""Testing module for the runners functions. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest

from bandpy import runners, env, controllers, agents
from bandpy.utils import pytest_set_up


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('early_stopping', [False, True])
def test_run_trials(d, early_stopping, seed):
    """Test the runner of 'run_trials'."""
    set_up = pytest_set_up(d=d, seed=seed)

    seed = set_up['seed']
    seeds = [seed] * 3

    N = 1

    bandit = env.ClusteredGaussianLinearBandit(N=N, T=10, d=d, K=2, n_thetas=1,
                                               seed=seed)

    agent_cls = agents.LinUniform
    agent_kwargs = dict(arms=bandit.arms, seed=seed)

    ctrl = controllers.Decentralized(agent_cls, agent_kwargs, N=N)

    trial_results = runners.run_trials(env=bandit,
                                       agent_or_controller=ctrl,
                                       early_stopping=early_stopping,
                                       seeds=seeds,
                                       n_jobs=1,
                                       verbose=False)

    assert len(seeds) == len(trial_results)
