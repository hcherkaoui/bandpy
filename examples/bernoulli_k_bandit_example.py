""" Simple example in Bandpy.

Launch it with ::
    $ python bernoulli_k_bandit_example.py --proba 0.1 0.2 0.1 0.1 0.5 0.3 \
                                           --n-trials 10 --T 200 --verbose

"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from bandpy import run_simulation, agents, env


plt.rcParams['text.usetex'] = True
DEFAULT_RESULT_DIR = '/mnt/c/Users/hwx1143141/Desktop/'


if __name__ == '__main__':

    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Number of trials.')
    parser.add_argument('--T', type=int, default=200,
                        help='Total number of iterations.')
    parser.add_argument('--proba', nargs="+", type=float, default=[0.1, 0.7],
                        help='List of probabilities for each arms.')
    parser.add_argument('--fig-dirname', type=str, default=DEFAULT_RESULT_DIR,
                        help='Regret evolution figure dirname.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbosity level.')
    args = parser.parse_args()

    ###########################################################################
    # Setting the simulation
    # Env. setting
    env_name = "Bernoulli-K-bandit"
    env_cls = env.BernoulliKBandit
    env_kwargs = {'p': args.proba, 'T': args.T, 'seed': args.seed}

    # Agents setting
    l_agent_cls = [agents.FollowTheLeader, agents.Uniform, agents.EC,
                   agents.UCB]
    l_agent_names = ["'Follow-the-leader' agent", "'Uniform' agent",
                     "'E-C' agent", "'UCB' agent"]
    l_agent_kwargs = [
            {'K': len(args.proba), 'seed': args.seed},
            {'K': len(args.proba), 'seed': args.seed},
            {'K': len(args.proba), 'm': 0.2, 'T': args.T, 'seed': args.seed},
            {'K': len(args.proba), 'delta': 0.01, 'seed': args.seed},
                     ]

    ###########################################################################
    # Running the simulation
    if args.verbose:
        print(f"[Main] Running agents: {l_agent_names}.")
        print(f"[Main] Env. '{env_name}' running with {len(args.proba)} arms "
              f"with probabilities={args.proba}.")

    all_regrets = run_simulation(args.n_trials, args.T, env_cls, env_kwargs,
                                 l_agent_names, l_agent_cls, l_agent_kwargs)

    ###########################################################################
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    for agent_name, regrets in all_regrets.items():
        mean_regrets = np.mean(regrets, axis=0)
        std_regrets = np.std(regrets, axis=0)

        ax.plot(np.arange(args.T), mean_regrets, lw=2.0,
                label=agent_name)
        ax.fill_between(np.arange(args.T),
                        mean_regrets + std_regrets / 2.0,
                        mean_regrets - std_regrets / 2.0,
                        alpha=0.3)

    fig.legend(ncol=2, loc='upper center', fontsize=12)
    ax.set_xlabel(r'$t$', fontsize=18)
    ax.set_ylabel(r'$R_t$', fontsize=18, rotation=0)
    ax.yaxis.set_label_coords(-0.1, 0.5)

    plt.grid()
    fig_fname = os.path.join(args.fig_dirname, 'regrets_evolution.png')
    fig.savefig(fig_fname, dpi=300)

    if args.verbose:
        print(f"[Main] Saving '{fig_fname}'.")

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    if args.verbose:
        print(f"[Main] Script runs in {delta_t}.")
