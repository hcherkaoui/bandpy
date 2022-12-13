"""Testing module for the runners functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest

from bandpy import run_trials, env, controller, agents
from bandpy.utils import tests_set_up


@pytest.mark.parametrize('d', [2, 10])
@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('controller_stop', [False, True])
def test_run_trials(d, controller_stop, seed):
    """Test the runner of 'run_trials'."""
    set_up = tests_set_up(d=d, seed=seed)

    seed = set_up['seed']
    seeds = [seed] * 3

    N = 1

    env_ = env.ClusteredGaussianLinearBandit(N=N, T=10, d=d, K=2, n_thetas=1,
                                             seed=seed)

    agent_cls = agents.LinUniform
    agent_kwargs = dict(arms=env_.arms, seed=seed)

    controller_ = controller.DecentralizedController(agent_cls, agent_kwargs,
                                                     N=N)

    trial_results = run_trials(env=env_, controller=controller_,
                               controller_stop=controller_stop, seeds=seeds,
                               n_jobs=1, verbose=False)

    assert len(seeds) == len(trial_results)
