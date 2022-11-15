""" Define plotting utility functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import matplotlib.pyplot as plt
import numpy as np


def plot_regrets(all_regrets, fig_fname=None, figsize_x=9, figsize_y=6,
                 verbose=False):
    """ Plot the regrets for all the environments and controllers."""

    env_names = list(all_regrets.keys())

    fig, axis = plt.subplots(nrows=len(env_names), ncols=1,
                             figsize=(figsize_x, figsize_y * len(env_names)))

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
                                label=f"R_t ({controller_name})")
            axis[axis_idx].fill_between(np.arange(T),
                                        mean_regrets + std_regrets,
                                        mean_regrets - std_regrets,
                                        alpha=0.3)

        axis[axis_idx].axhline(lw=1.5, color='black')
        axis[axis_idx].legend(ncol=2, loc='upper center', fontsize=13)
        axis[axis_idx].set_xlabel('t', fontsize=16)
        axis[axis_idx].set_title(f"Env: {env_name}", fontsize=16)
        axis[axis_idx].grid()

    fig.tight_layout()

    if fig_fname is not None:
        fig.savefig(fig_fname, dpi=300)

        if verbose:
            print(f"[Main] Saving '{fig_fname}'.")
    else:
        plt.plot()
