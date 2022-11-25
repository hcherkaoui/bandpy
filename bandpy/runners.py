""" Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import copy
import itertools
from joblib import Parallel, delayed
import numpy as np

from .utils import convert_grid_to_list


def run_one_trial(env,
                  controller,
                  controller_stop=False,
                  seed=None):
    """ Run on trial of 'controller' with environment 'env'.

    Parameters
    ----------
    env : env-class, environment instance.

    controller : controller-class, controller instance.

    controller_stop: bool, enable controller early-stopping.

    seed : None, int, random-instance, seed for the trial.

    Return
    ------
    T : int, number of iterations done.

    mean_regret : float, average regret for all agents.

    mean_reward : float, average cumulative reward for all agents.

    mean_best_reward : float, average cumulative best reward for all agents.

    mean_worst_reward : float, average cumulative worst reward for all agents.

    best_arms : dict, dictionary with agent_name as keys and index of the
        estimated best arm as values.

    controller : controller-class, controller class for later inspection.

    env : env-class, environment class for later inspection.
    """
    # ensure to reset the environment
    env.reset(seed=seed)

    # trigger first observations
    actions = controller.default_act()
    observations, rewards, _, _ = env.step(actions)

    while True:

        # controller/environment interaction
        actions = controller.act(observations, rewards)
        observations, rewards, done, _ = env.step(actions)

        # controller early-stopping
        if controller_stop & controller.done:
            break

        # environement early-stopping
        if done:
            break

    return controller, env


def run_trials(env,
               controller,
               controller_stop=False,
               seeds=None,
               n_jobs=1,
               verbose=True):
    """Run in parallel 'run_one_trial' with the given parameters.

    Parameters
    ----------
    env : env-class, environment instance.

    controller : controller-class, controller instance.

    controller_stop: bool, enable controller early-stopping.

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
            env=copy.deepcopy(env), controller=copy.deepcopy(controller),
            controller_stop=controller_stop,
            seed=seed)

        delayed_pool.append(delayed(run_one_trial)(**trial_kwargs))

    verbose_level = 100 if verbose else 0
    trial_results = Parallel(n_jobs=n_jobs,
                             verbose=verbose_level)(delayed_pool)

    return trial_results


def run_trials_with_grid_search(env_cls,
                                env_kwargs_grid,
                                controller_cls,
                                controller_kwargs_grid,
                                controller_stop=False,
                                seeds=None,
                                n_jobs_trials=1,
                                n_jobs_grid_search=1,
                                verbose=True):
    """Run 'run_trials' with a grid-search on the given parameters for both the
    environment and the controller.

    Parameters
    ----------
    env_cls : env-class, environment class.

    env_kwargs_grid : dict of list, grid for env parameters.

    controller_cls : controller-class, controller class.

    controller_kwargs_grid : dict of list, grid for controller parameters.

    controller_stop: bool, enable controller early-stopping.

    seed : None, int, random-instance, seed for the trial.

    n_jobs_trials : int, number of CPU to use for the trials.

    n_jobs_grid_search : int, number of CPU to use for the grid-search.

    verbose : bool, enable verbose.

    Return
    ------
    best_results : list, one-trial results, see 'run_one_trial' for more
        information.

    all_results : dict, dict of list (one-trial results) for inspection, see
        'run_one_trial' for more information.
    """
    env_kwargs_list = convert_grid_to_list(env_kwargs_grid)
    controller_kwargs_list = convert_grid_to_list(controller_kwargs_grid)

    envs = [env_cls(**env_kwargs) for env_kwargs in env_kwargs_list]
    controllers = [controller_cls(**controller_kwargs)
                   for controller_kwargs in controller_kwargs_list]

    def _run_trials(env, controller, seeds, n_jobs, verbose, controller_stop):
        trial_results = run_trials(env=env, controller=controller,
                                   controller_stop=controller_stop,
                                   seeds=seeds, n_jobs=n_jobs,
                                   verbose=verbose)
        l_last_regret = [trial_result[1][-1] for trial_result in trial_results]
        return np.mean(l_last_regret), trial_results

    delayed_pool = []
    for setting in itertools.product(envs, controllers):

        env, controller = setting

        trial_kwargs = dict(env=env, controller=controller,
                            controller_stop=controller_stop,
                            n_jobs=n_jobs_trials,
                            seeds=seeds, verbose=verbose)

        delayed_pool.append(delayed(_run_trials)(**trial_kwargs))

    verbose_level = 100 if verbose else 0
    all_results = Parallel(n_jobs=n_jobs_grid_search,
                           verbose=verbose_level)(delayed_pool)

    results = dict()
    for mean_last_regret, trial_results in all_results:
        results[mean_last_regret] = trial_results

    min_last_regret = np.min(list(results.keys()))

    return results[min_last_regret], results
