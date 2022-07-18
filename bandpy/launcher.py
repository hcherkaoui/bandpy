""" Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from joblib import Parallel, delayed
import numpy as np


def launch_experiment(env_names, envs, controller_names, controllers,
                      n_trials=10, n_jobs=1, seeds=None,
                      allow_early_stoppping=False, verbose=False):
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

    allow_early_stoppping : bool, default=False, to allow early stopping.

    verbose : bool, default=False, verbosity level.

    Return
    ------
    regrets : dict of dict of 2D np.array, regrets for each trial, for each
        environment times the controllers used.
    """
    if seeds is None:
        seeds = [None] * n_trials

    if len(seeds) != n_trials:
        raise ValueError(f"'seeds' should contains 'n_trials' seeds, got "
                         f"{len(seeds)}")

    all_results = {}  # either regrets or number of samples needed in BAI case

    for env_name, env in zip(env_names, envs):

        results = {}  # either regrets or number of samples needed in BAI case

        for controller_name, controller in zip(controller_names, controllers):

            if verbose:
                print(f"[run_simulation] running '{controller_name}' on "
                    f"'{env_name}'.")

            # run all the trial in parallel
            delayed_pool = []
            for seed in seeds:

                trial_kwargs = dict(
                            env=env, controller=controller, seed=seed,
                            allow_early_stoppping=allow_early_stoppping
                            )

                delayed_pool.append(delayed(run_one_trial)(**trial_kwargs))

            verbose_level = 100 if verbose else 0
            trial_results = Parallel(n_jobs=n_jobs,
                                     verbose=verbose_level)(delayed_pool)

            if allow_early_stoppping:
                trial_results = [len(trial_result)
                                 for trial_result in trial_results]

            results[controller_name] = np.array(trial_results)

        all_results[env_name] = results

    if verbose:
        print("[run_simulation] all run(s) done.")

    # either regrets or number of samples needed in BAI case
    return all_results
