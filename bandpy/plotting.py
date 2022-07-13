""" Define plotting utility functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_regrets(all_regrets, fig_dirname, verbose=False):
    """ Plot the regrets for all the environments and controllers."""

    env_names = list(all_regrets.keys())

    fig, axis = plt.subplots(nrows=len(env_names), ncols=1, sharex=True,
                             figsize=(9, 6 * len(env_names)))

    if len(env_names) == 1:
        axis = [axis]

    for axis_idx, env_name in enumerate(env_names):

        controller_names = list(all_regrets[env_name].keys())

        for controller_name in controller_names:

            regrets = all_regrets[env_name][controller_name]
            mean_regrets = np.mean(regrets, axis=0)
            std_regrets = np.std(regrets, axis=0)
            T = len(mean_regrets)

            axis[axis_idx].plot(np.arange(T), mean_regrets, lw=2.0,
                                label=controller_name)
            axis[axis_idx].fill_between(np.arange(T),
                                        mean_regrets + std_regrets / 2.0,
                                        mean_regrets - std_regrets / 2.0,
                                        alpha=0.3)

        axis[axis_idx].legend(ncol=2, loc='upper center', fontsize=12)
        axis[axis_idx].set_xlabel('t', fontsize=18)
        axis[axis_idx].set_ylabel('R_t', fontsize=18, rotation=0)
        axis[axis_idx].set_title(f"Environment: {env_name}", fontsize=18)
        axis[axis_idx].grid()

    fig_fname = os.path.join(fig_dirname, 'regrets_evolution.png')
    fig.savefig(fig_fname, dpi=300)

    if verbose:
        print(f"[Main] Saving '{fig_fname}'.")
