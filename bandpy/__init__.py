""" Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np


def run_simulation(n_trials, T, env_cls, env_kwargs, l_agent_names,
                   l_agent_cls, l_agent_kwargs):
    """ Run 'n_trials' time each agent in 'l_agent_cls' with 'env_cls'
    environment with an horizon 'T'.
    """
    all_regrets = {}
    for agent_name, agent_cls, agent_kwargs in zip(l_agent_names, l_agent_cls,
                                                   l_agent_kwargs):

        regrets = np.zeros((n_trials, T), dtype=float)
        for i in range(1, n_trials + 1):
            env = env_cls(**env_kwargs)
            agent = agent_cls(**agent_kwargs)

            # init. by pulling arm #0
            observation, reward, _, _ = env.step(0)

            while True:
                # agent/env iteration
                observation, reward, done, _ = env.step(
                                            agent.act(
                                                {'observation': observation,
                                                 'last_reward': reward}
                                                    )
                                                 )

                # regret computation
                regrets[i - 1, env.t - 1] = env.regret()

                if done:
                    break

        all_regrets[agent_name] = regrets

    return all_regrets
