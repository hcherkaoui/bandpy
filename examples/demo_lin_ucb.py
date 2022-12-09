""" Simple example with LinUCB policy.

Launch it with ::
    $ python demo_lin_ucb.py

"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.fr>

import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from bandpy import run_trials, env, controller, agents, utils

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 16
mpl.rcParams['text.latex.preamble'] = (r'\usepackage{amsmath}'
                                       r'\usepackage{amssymb}')


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
    parser.add_argument('--T', type=int, default=1000,
                        help='Number of iterations for the simulation.')
    parser.add_argument('--K', type=int, default=20,
                        help='Number of arms.')
    parser.add_argument('--d', type=int, default=20,
                        help='Dimension of the problem.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='UCB parameter.')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Standard deviation of the noise.')
    parser.add_argument('--te', type=int, default=10,
                        help='Number of iterations on which to exactly update'
                             'the inverse matrices.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of jobs to run in parallel the trials.')
    parser.add_argument('--fig-fname', type=str,
                        default='r__cum_r_evo.pdf',
                        help='Figure filename.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Setting the simulation
    if args.verbose:
        print("[main] Setting simulation.")

    bandit_env = env.ClusteredGaussianLinearBandit(
                                        N=args.N, T=args.T, d=args.d,
                                        K=args.K,
                                        n_thetas=1, sigma=args.sigma,
                                        shuffle_labels=False, seed=args.seed)

    agent_cls = agents.LinUCB
    agent_kwargs = {'alpha': args.alpha,
                    'arms': bandit_env.arms,
                    'te': args.te,
                    'seed': args.seed,
                    }

    bandit_controller = controller.DecentralizedController(
                    N=args.N, agent_cls=agent_cls, agent_kwargs=agent_kwargs)

    rng = utils.check_random_state(args.seed)
    seeds = rng.randint(MAX_RANDINT, size=args.n_trials)

    ###########################################################################
    # Run the simulation
    if args.verbose:
        print("[main] Running simulation.")

    trial_results = run_trials(bandit_env, bandit_controller,
                               controller_stop=False, seeds=seeds,
                               n_jobs=args.n_jobs, verbose=args.verbose)

    ###########################################################################
    # Gathering results
    if args.verbose:
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

    if args.verbose:
        print(f"[main] Saving plot under '{args.fig_fname}'.")

    plt.savefig(args.fig_fname, dpi=300)
