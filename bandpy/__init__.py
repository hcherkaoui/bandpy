""" Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from joblib import Parallel, delayed
import numpy as np


def run_trial(env, controller, enable_controller_early_stopping=False,
              seed=None):
    """ Run trial of 'controller' with environment 'env'.

    Parameters
    ----------
    env : env-class, environment.

    controller : controller-class, controller.

    enable_controller_early_stopping: bool, enable controller early-stopping.

    seed : None, int, random-instance, seed for the experiment.

    Return
    ------
    T : int, number of iteration.
    mean_regret : float, average regret for all agents.
    best_arms : dict, dictionary with agent_name as keys and idx of the
        estimated best arm as values.
    """
    # ensure to reset the environment
    env.reset(seed=seed)

    # trigger first observations
    actions = controller.init_act()
    observations, rewards, _, _ = env.step(actions)

    mean_regret, T = [], 0

    while True:

        # controller/environment interaction
        actions = controller.act(observations, rewards)
        observations, rewards, done, _ = env.step(actions)

        # regret storing
        all_agent_regrets = list(env.regret().values())
        all_agent_regrets = np.array(all_agent_regrets, dtype=float)
        mean_regret.append(np.mean(all_agent_regrets))

        # controller early-stopping
        if enable_controller_early_stopping & controller.done:
            break

        # environement early-stopping
        if done:
            break

        T += 1

    return T, mean_regret, controller.best_arms, controller, env


def run_trials(env, controller, enable_controller_early_stopping=False,
               seeds=None, n_jobs=1, verbose=True):
    """Run in parallel 'run_trial' with the given parameters."""
    delayed_pool = []
    for seed in seeds:
        trial_kwargs = dict(
            env=env, controller=controller,
            enable_controller_early_stopping=enable_controller_early_stopping,
            seed=seed,
                    )
        delayed_pool.append(delayed(run_trial)(**trial_kwargs))

    verbose_level = 100 if verbose else 0
    trial_results = Parallel(n_jobs=n_jobs,
                             verbose=verbose_level)(delayed_pool)

    return trial_results


def launch_experiment(env_names, envs, controller_names, controllers,
                      n_trials=10, n_jobs=1, seeds=None,
                      enable_controller_early_stopping=False, verbose=False):
    """ Run, for each env in 'envs', 'n_trials' time each controller in
    'controllers' with an horizon 'T'.

    Parameters
    ----------
    env_names : list of str, list of environment names.

    envs : list of env-class, list of environments.

    controller_names : list of str, list of controller names.

    controllers : list of controller-class, list of controllers.

    n_trials : int, default=10, number of trials.

    n_jobs : int, number of CPU to use.

    seeds : list of None, int, random-instance, seed for the experiment.

    enable_controller_early_stopping : bool, default=False, to allow early
        stopping.

    verbose : bool, default=False, verbosity level.

    Return
    ------
    regrets : dict of dict of list of tuple, the tuple gathers (T, regret,
        best_arm), the list entries correspond to each agent for each
        environment times the controllers used (the two dictionaries).
    """
    if seeds is None:
        seeds = [None] * n_trials

    if len(seeds) != n_trials:
        raise ValueError(f"'seeds' should contains 'n_trials' seeds, got "
                         f"{len(seeds)}")

    all_results = {}

    for env_name, env in zip(env_names, envs):

        results = {}

        for controller_name, controller in zip(controller_names, controllers):

            if verbose:
                print(f"[run_simulation] running '{controller_name}' on "
                      f"'{env_name}'.")

            trial_results = run_trials(env=env, controller=controller,
                                       enable_controller_early_stopping=enable_controller_early_stopping,  # noqa
                                       seeds=seeds, n_jobs=n_jobs,
                                       verbose=verbose)

            results[controller_name] = trial_results

        all_results[env_name] = results

    if verbose:
        print("[run_simulation] all run(s) done.")

    return all_results
