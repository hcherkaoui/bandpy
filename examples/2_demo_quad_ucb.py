""" Simple example with LinUCB policy.

Launch it with ::
    $ python 1_demo_lin_ucb.py

"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.fr>

import time
import matplotlib.pyplot as plt
import numpy as np
from bandpy import run_trials, env, controllers, agents, utils

plt.style.use('tableau-colorblind10')
MAX_RANDINT = 10000


###############################################################################
# main
if __name__ == '__main__':

    t0_total = time.time()

    ###########################################################################
    # Setting the simulation
    print("[main] Setting simulation.")

    N = 100
    T = 1000
    seed = None
    alpha = 1.0
    n_trials = 5

    bandit_env = env.ClusteredGaussianLinearBandit(N=N, T=1000, d=30, K=20,
                                                   n_thetas=1, sigma=1.0,
                                                   shuffle_labels=False,
                                                   seed=seed)

    agent_cls = agents.QuadUCB
    agent_kwargs = {'alpha': 1.0,
                    'arms': bandit_env.arms,
                    'te': int(T/10),
                    'seed': seed,
                    }

    bandit_controller = controllers.DecentralizedController(
                    N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs)

    rng = utils.check_random_state(seed)
    seeds = rng.randint(MAX_RANDINT, size=n_trials)

    ###########################################################################
    # Run the simulation
    print("[main] Running simulation.")

    trial_results = run_trials(bandit_env, bandit_controller,
                               early_stopping=False, seeds=seeds,
                               n_jobs=n_trials, verbose=False)

    ###########################################################################
    # Gathering results
    print("[main] Gathering results.")

    cumulative_regrets, instantaneous_regret = [], []
    for trial_result in trial_results:

        controller, env = trial_result

        instantaneous_regret.append(env.mean_instantaneous_regret())
        cumulative_regrets.append(np.cumsum(env.mean_instantaneous_regret()))

    ###########################################################################
    # Plotting
    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(9, 3), squeeze=False)

    # instantaneous regret
    mean_, std_, _, _, all_lengths = utils.tolerant_stats(instantaneous_regret)
    tt = np.arange(np.max(all_lengths))

    axis[0, 0].plot(tt, mean_, color='tab:blue', lw=3.0, linestyle='solid', )
    axis[0, 0].fill_between(tt, mean_ + std_, mean_ - std_, color='tab:blue',
                            alpha=0.2)

    axis[0, 0].set_xlabel(r'$t$')
    axis[0, 0].set_ylabel(r'$r_t$')

    axis[0, 0].set_xscale('log')

    axis[0, 0].set_ylim(0.0)

    axis[0, 0].grid()

    # cumulative instantaneous regret
    mean_, std_, _, _, _ = utils.tolerant_stats(cumulative_regrets)

    axis[0, 1].plot(tt, mean_, color='tab:blue', lw=3.0, linestyle='solid')
    axis[0, 1].fill_between(tt, mean_ + std_, mean_ - std_, color='tab:blue',
                            alpha=0.2)

    axis[0, 1].set_xlabel(r'$t$')
    axis[0, 1].set_ylabel(r'$R_t$')

    axis[0, 1].set_xscale('log')
    axis[0, 1].set_yscale('log')

    axis[0, 1].grid()

    fig.tight_layout()

    filename = '2_demo_quad_ucb.pdf'
    print(f"[main] Saving plot under '{filename}'.")
    plt.savefig(filename, dpi=300)

###############################################################################
# Runtime
delta_t = time.gmtime(time.time() - t0_total)
delta_t = time.strftime("%H h %M min %S s", delta_t)
print(f"[Main] Script runs in {delta_t}.")