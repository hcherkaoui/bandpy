""" Simple example in bandpy.
"""
import matplotlib.pyplot as plt
import numpy as np
from bandpy.env import BernoulliKBandit
from bandpy.agents import FollowTheLeader


###############################################################################
# Globals
# simulation parameters
n_try = 50

# env parameters
horizon = 300
p = [0.1, 0.5, 0.3]
seed = None
K = len(p)

# store regrets evolution
regrets = np.empty((n_try, horizon), dtype=float)

###############################################################################
# Simulation
for i in range(1, n_try + 1):
    bk_bandit = BernoulliKBandit(p=p, seed=seed)
    agent = FollowTheLeader(K=K, seed=seed)

    for n in range(1, horizon + 1):

        k = agent.choose_arm()
        reward = bk_bandit.pull(k)
        agent.observe_env(k, reward)
        regret = bk_bandit.regret()

        regrets[i - 1, n - 1] = regret / float(n)

###############################################################################
# Plotting
plt.figure(figsize=(6, 4))
mean_regrets = np.mean(regrets, axis=0)
std_regrets = np.std(regrets, axis=0)
plt.plot(np.arange(horizon), mean_regrets, lw=1.5)
plt.fill_between(np.arange(horizon),
                 mean_regrets + std_regrets / 2.,
                 mean_regrets - std_regrets / 2.,
                 alpha=0.1)
plt.grid()
plt.xlabel("n", fontsize=12)
plt.ylabel("Regret", fontsize=12)
plt.savefig("/mnt/c/Users/hwx1143141/Desktop/regret_evolution.png")
