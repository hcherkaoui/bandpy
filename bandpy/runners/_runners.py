""" Define all the runner (without grid-search) functions availables. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import copy
from joblib import Parallel, delayed

from ..controllers._controllers import SingleAgentController
from ..agents._base import MultiLinearAgentsBase, SingleMABAgentBase


valid_agents_class = (
    MultiLinearAgentsBase,  # linear agent
    SingleMABAgentBase,  # vanilla agent
)


def run_one_trial(env, agent_or_controller, early_stopping=False, seed=None):
    """Run on trial of 'agent_or_controller' with environment 'env'.

    Parameters
    ----------
    env : env-class, environment instance.

    agent_or_controller : agent-class or controller-class, agent/controller
        instance.

    early_stopping: bool, enable controller early-stopping.

    seed : None, int, random-instance, seed for the trial.

    Return
    ------
    controller : controller-class, controller class for later inspection.

    env : env-class, environment class for later inspection.
    """
    # ensure to have a controller (as a proxy at least)
    if issubclass(type(agent_or_controller), valid_agents_class):
        controller = SingleAgentController(agent_or_controller)
    else:
        controller = agent_or_controller

    # ensure to reset the environment
    env.reset(seed=seed)

    # trigger first observations
    action = controller.default_act()
    observation, reward, _, info = env.step(action)

    while True:
        # controller/environment interaction
        action = controller.act(observation, reward, info)
        observation, reward, done, info = env.step(action)

        # controller early-stopping
        if early_stopping and controller.done:
            break

        # environment early-stopping
        if done:
            break

    return controller, env


def run_trials(env, agent_or_controller, early_stopping=False, seeds=None, n_jobs=1, verbose=True):  # noqa
    """Run in parallel 'run_one_trial' with the given parameters.

    Parameters
    ----------
    env : env-class, environment instance.

    agent_or_controller : agent-class or controller-class, agent/controller
        instance.

    early_stopping: bool, enable controller early-stopping.

    seed : None, int, random-instance, seed for the trial.

    n_jobs : int, number of CPU to use.

    verbose : bool, enable verbose.

    Return
    ------
    results : list, one-trial results, see 'run_one_trial' for more
        information.
    """
    delayed_pool = []
    for seed in seeds:
        trial_kwargs = dict(
            env=copy.deepcopy(env),
            agent_or_controller=copy.deepcopy(agent_or_controller),
            early_stopping=early_stopping,
            seed=seed,
        )

        delayed_pool.append(delayed(run_one_trial)(**trial_kwargs))

    verbose_level = 100 if verbose else 0
    trial_results = Parallel(n_jobs=n_jobs, verbose=verbose_level)(delayed_pool)

    return trial_results
