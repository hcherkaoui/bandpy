""" Simple benchmark in Bandpy.

Launch it with ::
    $ python benchmark.py --verbose

"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from bandpy import run_simulation, agents, env, controllers, plotting


plt.style.use('tableau-colorblind10')
DEFAULT_RESULT_DIR = '/mnt/c/Users/hwx1143141/Desktop/'


if __name__ == '__main__':

    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Number of trials.')
    parser.add_argument('--T', type=int, default=200,
                        help='Total number of iterations.')
    parser.add_argument('--N', type=int, default=10,
                        help='Total number of agents.')
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
    # Environment definition
    env_names = ["Bernoulli-K-bandit", "Gaussian-K-Bandit", "Linear-Bandit-2D"]
    envs = [env.BernoulliKBandit(p=[0.1, 0.5, 0.6], T=args.T, seed=args.seed),
            env.GaussianKBandit(mu=[0.1, 0.5, 0.6], sigma=[1.0, 1.0, 1.0], T=args.T, seed=args.seed),
            env.LinearBandit2D(delta=0.01, T=args.T, seed=args.seed),
            ]

    # Controller definition
    controller_names = ["Follow-the-leader agent", "Uniform agent", "E-C agent", "UCB agent"]
    controllers = [controllers.DecentralizedController(N=args.N, agent_cls=agents.FollowTheLeader, agent_kwargs={'K': 3, 'seed': args.seed}),
                   controllers.DecentralizedController(N=args.N, agent_cls=agents.Uniform, agent_kwargs={'K': 3, 'seed': args.seed}),
                   controllers.DecentralizedController(N=args.N, agent_cls=agents.EC, agent_kwargs={'K': 3, 'm': 0.2, 'T': args.T, 'seed': args.seed}),
                   controllers.DecentralizedController(N=args.N, agent_cls=agents.UCB, agent_kwargs={'K': 3, 'delta': 0.01, 'seed': args.seed}),
                   ]

    ###########################################################################
    # Running the simulation
    if args.verbose:
        print(f"[Main] Running controllers: {controller_names}.")
        print(f"[Main] Environment: '{env_names}'.")

    simulation_parameters = dict(env_names=env_names, envs=envs,
                                 controller_names=controller_names,
                                 controllers=controllers,
                                 n_trials=args.n_trials, T=args.T,
                                 verbose=args.verbose)
    all_regrets = run_simulation(**simulation_parameters)

    ###########################################################################
    # Plotting
    plotting.plot_regrets(all_regrets, args.fig_dirname, verbose=args.verbose)

    ###########################################################################
    # Runtime
    delta_t = time.gmtime(time.time() - t0_total)
    delta_t = time.strftime("%H h %M min %S s", delta_t)
    if args.verbose:
        print(f"[Main] Script runs in {delta_t}.")
