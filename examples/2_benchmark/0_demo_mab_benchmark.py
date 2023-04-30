""" Simple example with UCB policy.

Launch it with ::
    $ python 0_demo_benchmark.py --n-jobs 5 --verbose

"""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.fr>

import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from bandpy import runners, env, controllers, agents, utils


plt.style.use("tableau-colorblind10")
MAX_RANDINT = 10000
fontsize = 14

agent_colors = {
    r"UCB": "tab:orange",
    r"Uniform": "tab:gray",
    r"FollowTheLeader": "tab:blue",
    r"EC": "tab:green",
}

agent_linestyles = {
    r"UCB": "solid",
    r"Uniform": "dashed",
    r"FollowTheLeader": "solid",
    r"EC": "solid",
}


###############################################################################
# main
if __name__ == "__main__":
    t0_total = time.time()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=1, help="Number of trials.")
    parser.add_argument("--N", type=int, default=100, help="Number of agents.")
    parser.add_argument(
        "--T", type=int, default=50000, help="Number of iterations for the simulation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed for the experiment. Can be used "
        "for debug or to freeze experiments.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel the trials.",
    )
    parser.add_argument(
        "--fig-fname",
        type=str,
        default="0_demo_mab_benchmark.pdf",
        help="Figure filename.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbosity level.")
    args = parser.parse_args()

    ###########################################################################
    # Setting the simulation
    if args.verbose:
        print("[Main] Setting simulation")

    mu = [1.0, 0.5, 2.5]
    sigma = [1.0, 2.0, 0.1]

    bandit_env = env.GaussianKBandit(mu=mu, sigma=sigma, T=args.T, seed=args.seed)

    l_agent_kits = [
        {
            "agent_name": "UCB",
            "agent_cls": agents.UCB,
            "agent_kwargs": {
                "delta": 0.1,
                "K": bandit_env.K,
                "seed": args.seed,
            },
        },
        {
            "agent_name": "Uniform",
            "agent_cls": agents.Uniform,
            "agent_kwargs": {
                "K": bandit_env.K,
                "seed": args.seed,
            },
        },
        {
            "agent_name": "FollowTheLeader",
            "agent_cls": agents.FollowTheLeader,
            "agent_kwargs": {
                "K": bandit_env.K,
                "seed": args.seed,
            },
        },
        {
            "agent_name": "EC",
            "agent_cls": agents.EC,
            "agent_kwargs": {
                "Te": int(args.T / 3),
                "K": bandit_env.K,
                "seed": args.seed,
            },
        },
    ]

    ###########################################################################
    # Run the simulation
    trial_results = dict()
    for agent_kits in l_agent_kits:
        agent_name = agent_kits["agent_name"]
        agent_cls = agent_kits["agent_cls"]
        agent_kwargs = agent_kits["agent_kwargs"]

        bandit_controller = controllers.Decentralized(
            N=args.N, agent_cls=agent_cls, agent_kwargs=agent_kwargs
        )

        rng = utils.check_random_state(args.seed)
        seeds = rng.randint(MAX_RANDINT, size=args.n_trials)

        if args.verbose:
            print(f"[Main] Running simulation with agent '{agent_name}'")

        trial_results[agent_name] = runners.run_trials(
            bandit_env,
            bandit_controller,
            early_stopping=False,
            seeds=seeds,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )

    ###########################################################################
    # Gathering results
    if args.verbose:
        print("[Main] Gathering results")

    no_noise_R_t = dict()
    for agent_name, trial_results_per_agent in trial_results.items():
        l_no_noise_R_t = []
        for one_trial_result in trial_results_per_agent:
            _, bandit_env = one_trial_result

            l_no_noise_R_t.append(bandit_env.no_noise_R_t)  # noqa

        no_noise_R_t[agent_name] = l_no_noise_R_t

    ###########################################################################
    # Plotting
    plt.figure(figsize=(6, 3))

    for agent_name in no_noise_R_t.keys():
        mean_, std_, _, _, all_lengths = utils.tolerant_stats(
            no_noise_R_t[agent_name]
        )  # noqa
        tt = np.arange(np.max(all_lengths))

        plt.plot(
            tt,
            mean_,
            color=agent_colors[agent_name],
            lw=1.0,
            linestyle=agent_linestyles[agent_name],
            label=agent_name,
        )
        plt.fill_between(
            tt, mean_ + std_, mean_ - std_, color=agent_colors[agent_name], alpha=0.3
        )

    plt.xlabel(r"$t$", fontsize=fontsize)
    plt.ylabel(r"$R_t$", fontsize=fontsize)

    plt.grid()
    plt.legend(ncol=1, fontsize=fontsize, loc="center left", bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()

    if args.verbose:
        print(f"[Main] Saving plot under '{args.fig_fname}'")

    plt.savefig(args.fig_fname, dpi=300)

###############################################################################
# Runtime
delta_t = time.gmtime(time.time() - t0_total)
delta_t = time.strftime("%H h %M min %S s", delta_t)
if args.verbose:
    print(f"[Main] Script runs in {delta_t}")
