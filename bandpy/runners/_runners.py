""" Define all the runner (without grid-search) functions availables. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import copy
from joblib import Parallel, delayed


def run_one_trial(env, agent_or_controller, early_stopping=False, seed=None):
    """ Run on trial of 'agent_or_controller' with environment 'env'.

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
    # ensure to reset the environment
    env.reset(seed=seed)

    # trigger first observations
    action = agent_or_controller.default_act()
    observation, reward, _, _ = env.step(action)

    while True:

        # controller/environment interaction
        action = agent_or_controller.act(observation, reward)
        observation, reward, done, _ = env.step(action)

        # early-stopping
        if early_stopping & agent_or_controller.done:
            break

        # environement early-stopping
        if done:
            break

    return agent_or_controller, env


def run_trials(env,
               agent_or_controller,
               early_stopping=False,
               seeds=None,
               n_jobs=1,
               verbose=True):
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
                    seed=seed)

        delayed_pool.append(delayed(run_one_trial)(**trial_kwargs))

    verbose_level = 100 if verbose else 0
    trial_results = Parallel(n_jobs=n_jobs,
                             verbose=verbose_level)(delayed_pool)

    return trial_results
