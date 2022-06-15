""" Simple example in Bandpy.

Launch it with ::
    $ python simple_example.py --proba 0.1 0.7 0.3 --n-trials 50 \
                               --horizon 500 --verbose

"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.fr>

import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from bandpy.env import BernoulliKBandit
from bandpy.agents import FollowTheLeader, Uniform


plt.rcParams['text.usetex'] = True
DEFAULT_RESULT_DIR = '/mnt/c/Users/hwx1143141/Desktop/'


if __name__ == '__main__':

    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of trials.')
    parser.add_argument('--horizon', type=int, default=500,
                        help='Number of iterations.')
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
    agent_type = [FollowTheLeader, Uniform]
    agent_name = ["'Follow-the-leader' agent", "'Uniform' agent"]

    if args.verbose:
        print(f"[Main] Running agents: {agent_name}.")
        print(f"[Main] Env. running with {len(args.proba)} arms with "
              f"probabilities={args.proba}.")

    ###########################################################################
    # Running the simulation
    all_regrets = {}
    for agent_name, agent_type in zip(agent_name, agent_type):

        regrets = np.empty((args.n_trials, args.horizon), dtype=float)
        for i in range(1, args.n_trials + 1):
            env = BernoulliKBandit(p=args.proba, horizon=args.horizon,
                                   seed=args.seed)
            agent = agent_type(K=len(args.proba), seed=args.seed)

            observation, reward, _, _ = env.step(0)  # init. by pulling arm #0
            kwargs = {'observation': observation, 'last_reward': reward}
            while True:

                # agent/env iteration
                observation, reward, done, info = env.step(
                                            agent.act(
                                                {'observation': observation,
                                                 'last_reward': reward}))

                # regret computation
                regret = np.max(env.p) - env.total_reward / env.n
                regrets[i - 1, env.n - 1] = regret

                if done:
                    break

        all_regrets[agent_name] = regrets

    ###########################################################################
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    for agent_name, regrets in all_regrets.items():
        mean_regrets = np.mean(regrets, axis=0)
        std_regrets = np.std(regrets, axis=0)

        ax.plot(np.arange(args.horizon), mean_regrets, lw=2.0,
                label=agent_name)
        ax.fill_between(np.arange(args.horizon),
                        mean_regrets + std_regrets / 2.0,
                        mean_regrets - std_regrets / 2.0,
                        alpha=0.3)

    fig.legend(ncol=len(agent_name), loc='upper center', fontsize=12)
    ax.set_xlabel(r'$n$', fontsize=18)
    ax.set_ylabel(r'$R_n$', fontsize=18, rotation=0)
    ax.yaxis.set_label_coords(-0.1, 0.5)

    plt.grid()
    fig_fname = os.path.join(args.fig_dirname, 'regret_evolution.png')
    fig.savefig(fig_fname, dpi=300)

    if args.verbose:
        print(f"[Main] Saving '{fig_fname}'.")

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    if args.verbose:
        print(f"[Main] Script runs in {delta_t}.")
