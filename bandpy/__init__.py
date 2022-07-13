""" Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np


def run_simulation(env_names, envs, controller_names, controllers, n_trials=10,
                   T=500, verbose=False):
    """ Run, for each env in 'envs', 'n_trials' time each controller in
    'controllers' with an horizon 'T'.

    Parameters
    ----------
    env_names : str,

    envs : env-class,

    controller_names : str,

    controllers : controller-class,

    n_trials : int, default=10

    T : int, default=500

    verbose : bool, default=False


    Return
    ------
    regrets :
    """
    all_regrets = {}

    for env_name, env in zip(env_names, envs):

        regrets = {}

        for controller_name, controller in zip(controller_names, controllers):

            if verbose:
                print(f"[run_simulation] running '{controller_name}' on "
                      f"'{env_name}'.")

            trial_regrets = np.zeros((n_trials, T), dtype=float)

            for i in range(1, n_trials + 1):

                # reset environment
                env.reset()

                # trigger first observations
                actions = controller.init()
                observations, rewards, _, _ = env.step(actions)

                while True:

                    # controller/environment interaction
                    actions = controller.act(observations, rewards)
                    observations, rewards, done, _ = env.step(actions)

                    # regret storing
                    mean_trial_regret = np.mean(list(env.regret().values()))
                    trial_regrets[i - 1, env.t - 1] = mean_trial_regret

                    # simulation early-stopping
                    if done:
                        break

            regrets[controller_name] = trial_regrets

        all_regrets[env_name] = regrets

    if verbose:
        print("[run_simulation] all runs done.")

    return all_regrets
