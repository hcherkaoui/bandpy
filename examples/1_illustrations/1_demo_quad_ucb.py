""" Simple example with QuadUCB policy.

Launch it with ::
    $ python 2_demo_quad_ucb.py --n-jobs 5 --verbose

"""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.fr>

import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from bandpy import runners, env, controllers, agents, utils

plt.style.use('tableau-colorblind10')
MAX_RANDINT = 10000


###############################################################################
# main
if __name__ == '__main__':

    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-trials', type=int, default=1,
                        help='Number of trials.')
    parser.add_argument('--N', type=int, default=100,
                        help='Number of agents.')
    parser.add_argument('--T', type=int, default=50000,
                        help='Number of iterations for the simulation.')
    parser.add_argument('--K', type=int, default=20,
                        help='Number of arms.')
    parser.add_argument('--d', type=int, default=20,
                        help='Dimension of the problem.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='UCB parameter.')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Standard deviation of the noise.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of jobs to run in parallel the trials.')
    parser.add_argument('--fig-fname', type=str,
                        default='2_demo_quad_ucb.pdf',
                        help='Figure filename.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Setting the simulation
    if args.verbose:
        print("[Main] Setting simulation")

    bandit_env = env.ClusteredGaussianLinearBandit(
                                        N=args.N, T=args.T, d=args.d,
                                        K=args.K, n_thetas=1, sigma=args.sigma,
                                        shuffle_labels=False, seed=args.seed)

    agent_cls = agents.QuadUCB
    agent_kwargs = {'alpha': args.alpha,
                    'arms': bandit_env.arms,
                    'seed': args.seed,
                    }

    bandit_controller = controllers.Decentralized(
                    N=args.N, agent_cls=agent_cls, agent_kwargs=agent_kwargs)

    rng = utils.check_random_state(args.seed)
    seeds = rng.randint(MAX_RANDINT, size=args.n_trials)

    ###########################################################################
    # Run the simulation
    if args.verbose:
        print("[Main] Running simulation")

    trial_results = runners.run_trials(
                            bandit_env, bandit_controller,
                            early_stopping=False, seeds=seeds,
                            n_jobs=args.n_jobs, verbose=args.verbose,
                            )

    ###########################################################################
    # Gathering results
    if args.verbose:
        print("[Main] Gathering results")

    no_noise_r_t, no_noise_R_t = [], []
    for trial_result in trial_results:

        _, bandit_env = trial_result

        no_noise_r_t.append(bandit_env.no_noise_r_t)
        no_noise_R_t.append(bandit_env.no_noise_R_t)

    ###########################################################################
    # Plotting
    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), squeeze=False)

    # instantaneous regret
    mean_, std_, _, _, all_lengths = utils.tolerant_stats(no_noise_r_t)
    tt = np.arange(np.max(all_lengths))

    axis[0, 0].plot(tt, mean_, color='tab:blue', lw=1.0, linestyle='solid', )
    axis[0, 0].fill_between(tt, mean_ + std_, mean_ - std_, color='tab:blue',
                            alpha=0.2)

    axis[0, 0].set_xlabel(r'$t$')
    axis[0, 0].set_ylabel(r'$r_t$')

    axis[0, 0].set_xscale('log')

    axis[0, 0].set_ylim(0.0)

    axis[0, 0].grid()

    # cumulative instantaneous regret
    mean_, std_, _, _, _ = utils.tolerant_stats(no_noise_R_t)

    axis[0, 1].plot(tt, mean_, color='tab:blue', lw=1.0, linestyle='solid')
    axis[0, 1].fill_between(tt, mean_ + std_, mean_ - std_, color='tab:blue',
                            alpha=0.2)

    axis[0, 1].set_xlabel(r'$t$')
    axis[0, 1].set_ylabel(r'$R_t$')

    axis[0, 1].set_xscale('log')
    axis[0, 1].set_yscale('log')

    axis[0, 1].grid()

    fig.tight_layout()

    if args.verbose:
        print(f"[Main] Saving plot under '{args.fig_fname}'")

    plt.savefig(args.fig_fname, dpi=300)

###############################################################################
# Runtime
delta_t = time.gmtime(time.time() - t0_total)
delta_t = time.strftime("%H h %M min %S s", delta_t)
if args.verbose:
    print(f"[Main] Script runs in {delta_t}")