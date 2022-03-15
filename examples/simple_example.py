""" Simple example in bandpy.
"""
import matplotlib.pyplot as plt
import numpy as np
from bandpy.env import BernoulliKBandit
from bandpy.agents import FollowTheLeader


plt.rcParams['text.usetex'] = True


###############################################################################
# Globals
# simulation parameters
n_try = 50

# env parameters
horizon = 200
p = [0.1, 0.6, 0.3]
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
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

mean_regrets = np.mean(regrets, axis=0)
std_regrets = np.std(regrets, axis=0)

ax.plot(np.arange(horizon), mean_regrets, lw=2.0)
ax.fill_between(np.arange(horizon),
                mean_regrets + std_regrets / 2.0,
                mean_regrets - std_regrets / 2.0,
                alpha=0.25)
ax.set_xlabel(r'$n$', fontsize=17)
ax.set_ylabel(r'$R_n$', fontsize=17, rotation=0)
ax.yaxis.set_label_coords(-0.1, 0.5)

plt.grid()
plt.savefig("/mnt/c/Users/hwx1143141/Desktop/regret_evolution.png", dpi=200)
