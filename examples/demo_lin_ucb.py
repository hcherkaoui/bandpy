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
    parser.add_argument('--n-trials', type=int, default=5,
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
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of jobs to run in parallel the trials.')
    parser.add_argument('--fig-fname', type=str,
                        default='regret_and_reward_evolution.pdf',
                        help='Figure filename.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbosity level.')
    args = parser.parse_args()

###############################################################################
# Setting the simulation
bandit_env = env.ClusteredGaussianLinearBandit(
                                N=args.N, T=args.T, d=args.d, K=args.K,
                                n_thetas=1, sigma=args.sigma,
                                shuffle_labels=False, seed=args.seed)

bandit_controller = controller.DecentralizedController(
                        N=args.N, agent_cls=agents.LinUCB,
                        agent_kwargs={'arms': bandit_env.arms,
                                      'alpha': args.alpha})

rng = utils.check_random_state(args.seed)
seeds = rng.randint(MAX_RANDINT, size=args.n_trials)

###############################################################################
# Run the simulation
trial_results = run_trials(bandit_env, bandit_controller,
                           controller_stop=False, seeds=seeds,
                           n_jobs=args.n_jobs, verbose=args.verbose)

###############################################################################
# Gathering results
rewards, best_rewards, regrets = [], [], []
for trial_result in trial_results:

    _, mean_regret, mean_reward, mean_best_reward, _, _, _, _ = trial_result

    regrets.append(mean_regret)
    rewards.append(mean_reward)
    best_rewards.append(mean_best_reward)

###############################################################################
# Plotting

fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(9, 3), squeeze=False)

# regret
mean_, std_, _, _, all_lengths = utils.tolerant_stats(regrets)
tt = np.arange(np.max(all_lengths))

axis[0, 0].plot(tt, mean_, color='tab:blue', lw=3.0, linestyle='solid', )
axis[0, 0].fill_between(tt, mean_ + std_, mean_ - std_, color='tab:blue',
                        alpha=0.2)

axis[0, 0].set_xlabel('t', fontsize=16)
axis[0, 0].set_ylabel('Cumulative regret')
axis[0, 0].set_xscale('log')
axis[0, 0].set_yscale('log')

axis[0, 0].grid()

# cumulative reward
mean_rewards, std_rewards, _, _, _ = utils.tolerant_stats(rewards)
mean_best_rewards, _, _, _, _ = utils.tolerant_stats(best_rewards)

axis[0, 1].plot(tt, mean_rewards, color='tab:blue', lw=3.0, linestyle='solid',
                label=r'S_t')
axis[0, 1].fill_between(tt, mean_rewards + std_rewards,
                        mean_rewards - std_rewards, color='tab:blue',
                        alpha=0.2)
axis[0, 1].plot(tt, mean_best_rewards, color='tab:red', lw=3.0,
                linestyle='dashed',label=r'S_t$^{\text{max}}$')

axis[0, 1].legend(loc='center left', bbox_to_anchor=(1.0, 0.6), ncol=1)

axis[0, 1].set_xlabel('t')
axis[0, 1].set_ylabel('Cumulative rewards')
axis[0, 1].set_xscale('log')
axis[0, 1].set_yscale('log')

axis[0, 1].grid()

fig.tight_layout()

plt.savefig(args.fig_fname, dpi=300)
