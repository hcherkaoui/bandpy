""" Simple example to benchmark controllers without clustering process

Launch it with ::
    $ python 1_demo_lb_benchmark_no_clustering.py

"""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.fr>

import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from bandpy import runners, env, controllers, agents, utils


plt.style.use('tableau-colorblind10')
max_randint = 10000
n_thetas = 2
tiny_angle = np.pi / 16.0
angle = np.pi * 7.0 / 8.0


###############################################################################
# main
if __name__ == '__main__':

    t0_total = time.perf_counter()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-trials', type=int, default=1,
                        help='Number of trials.')
    parser.add_argument('--T', type=int, default=50000,
                        help='Number of iterations for the simulation.')
    parser.add_argument('--K', type=int, default=20,
                        help='Number of arms.')
    parser.add_argument('--d', type=int, default=20,
                        help='Dimension of the problem.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='UCB parameter.')
    parser.add_argument('--lbda', type=float, default=1.0,
                        help='Ridge parameter.')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Standard deviation of the noise.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set the seed for the experiment. Can be used '
                        'for debug or to freeze experiments.')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of jobs to run in parallel the trials.')
    parser.add_argument('--fig-fname', type=str,
                        default='1_demo_lb_benchmark_no_clustering.pdf',
                        help='Figure filename.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbosity level.')
    args = parser.parse_args()

    rng = utils.check_random_state(args.seed)
    A_init = args.lbda * np.eye(args.d)

    all_alpha = [0.1, 1.0, 5.0]
    all_N = [n_thetas, 18]
    all_results = dict()
    for N in all_N:
        for alpha in all_alpha:

            true_labels = []
            for l in range(n_thetas):
                true_labels += [l] * int(N / n_thetas)
            true_labels += [l] * (N - len(true_labels))

            true_agent_labels = dict()
            for i, l in enumerate(true_labels):
                true_agent_labels[f"agent_{i}"] = l

            if args.verbose:
                print("=" * 80)
                print(f"[Main]: alpha = {alpha:.2f}, N = {N}.")

            thetas, arms = utils.generate_thetas_arms_gaussian(
                args.K, args.d, n_thetas, angle, tiny_angle, rng)

            S = np.max([np.linalg.norm(theta) for theta in thetas])

            env_instance = env.ClusteredGaussianLinearBandit(
                            d=args.d, N=N, T=args.T, arms=arms, thetas=thetas,
                            theta_idx=true_labels, sigma=args.sigma,
                            shuffle_labels=False, theta_offset=0.0,
                            seed=args.seed)

            all_controllers = {
            '0-Single': (controllers.SingleCluster,
                       {'N': N,
                        'agent_cls': agents.LinUCB,
                        'agent_kwargs': dict(arms=arms, alpha=alpha,
                                             lbda=args.lbda, seed=args.seed),
                        'seed': args.seed,
                        }),
            '1-Ind': (controllers.Decentralized,
                    {'N': N,
                     'agent_cls': agents.LinUCB,
                     'agent_kwargs': dict(arms=arms, alpha=alpha,
                                          lbda=args.lbda, seed=args.seed),
                     'seed': args.seed,
                     }),
            '2-Oracle': (controllers.OracleClustering,
                       {'N': N,
                        'agent_cls': agents.LinUCB,
                        'agent_kwargs': dict(arms=arms, alpha=alpha,
                                             lbda=args.lbda, seed=args.seed),
                        'agent_labels': true_agent_labels,
                        'seed': args.seed,
                        }),
            }

            for iterates in all_controllers.items():

                controller_name, (controller_cls, controller_kwargs) = iterates

                controller_instance = controller_cls(**controller_kwargs)

                if args.verbose:
                    print(f"[Main]: Running {controller_name}.")

                t0 = time.perf_counter()

                seeds = rng.randint(max_randint, size=args.n_trials)

                results = runners.run_trials(
                                     env=env_instance,
                                     agent_or_controller=controller_instance,
                                     early_stopping=False,
                                     seeds=seeds,
                                     n_jobs=args.n_jobs,
                                     verbose=args.verbose,
                                     )

                all_results[(alpha, N, controller_name)] = results

                mean_no_noise_R_T = np.mean([env_instance.no_noise_R_t[-1]
                                             for (_, env_instance) in results])

                delta_t = time.perf_counter() - t0
                if args.verbose:
                    print(f"[Main]: Finished with RT = {mean_no_noise_R_T:.2f}"
                          f" (runtime: {utils.format_duration(delta_t)}).")
                    print("-" * 80)

    ###########################################################################
    # Plotting options
    nb_points_to_plot = int(args.T / 10.0)
    start_plotting = 0
    stop_plotting = args.T
    step_plotting = int(args.T / nb_points_to_plot)
    tt = np.arange(start=start_plotting, stop=stop_plotting,
                   step=step_plotting)
    idx_slice = slice(start_plotting, stop_plotting, step_plotting)

    controller_colors = {
        '0-Single': 'lightgray',
        '1-Ind': 'gray',
        '2-Oracle': 'red',
    }

    controller_linestyles = {
        '0-Single': 'dashed',
        '1-Ind': 'dashed',
        '2-Oracle': 'dashed',
    }

    ###########################################################################
    # Plotting
    best_results = dict()
    for (alpha, N, controller_name), results in all_results.items():

        if not (N, controller_name) in best_results:
            best_results[(N, controller_name)] = results

        else:
            new_mean_R_T = np.mean([result[1].no_noise_R_t[-1]
                                    for result in results])
            ref_mean_R_T = np.mean([result[1].no_noise_R_t[-1]
                                    for result in best_results[(N, controller_name)]])  # noqa

            if new_mean_R_T < ref_mean_R_T:
                best_results[(N, controller_name)] = results

    fig, axis = plt.subplots(nrows=1,
                             ncols=len(all_N),
                             figsize=(len(all_N) * 4.0, 4.0),
                             squeeze=False)

    cols_idx = dict([(N, j) for j, N in enumerate(all_N)])
    all_mean_no_noise_R_T = []
    for (N, controller_name), results in best_results.items():

        j = cols_idx[N]

        l_no_noise_R_t = [env_instance.no_noise_R_t[idx_slice]
                          for (controller_instance, env_instance) in results]

        mean_no_noise_R_t = utils.tolerant_mean(l_no_noise_R_t)
        all_mean_no_noise_R_T.append(mean_no_noise_R_t[-1])

        axis[0, j].plot(tt,
                        mean_no_noise_R_t,
                        lw=4.0,
                        color=controller_colors[controller_name],
                        linestyle=controller_linestyles[controller_name],
                        label=controller_name,
                        alpha=0.5,
                        )

        axis[0, j].grid()
        axis[0, j].legend(ncol=1, fontsize=12)
        axis[0, j].set_xlabel("t", fontsize=14)
        axis[0, j].set_ylabel("Rt", fontsize=14)
        axis[0, j].set_title(f"N = {N}", fontsize=14)

    for j in range(len(all_N)):
        axis[0, j].set_ylim(bottom=0.0, top=np.max(all_mean_no_noise_R_T))

    fig.tight_layout()

    if args.verbose:
        print(f"[Main] Saving plot under '{args.fig_fname}'")

    plt.savefig(args.fig_fname, dpi=300)

    ###########################################################################
    # Runtime
    delta_t = time.perf_counter() - t0_total
    if args.verbose:
        print(f"[Main] Script runs in {utils.format_duration(delta_t)}")
