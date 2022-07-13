""" Define all the controllers availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from .base import Controller


class DecentralizedController(Controller):
    """DecentralizedController class to define a simple decentralized
    multi-agents setting.
    """

    def __init__(self, N, agent_cls, agent_kwargs):
        """Init."""
        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs)

    def act(self, observations, rewards):
        """Make each agent choose an arm in a decentralized way."""
        actions = dict()
        for agent_name in observations.keys():

            observation = observations[agent_name]
            reward = rewards[agent_name]

            k = self.agents[agent_name].act(observation, reward)
            actions[agent_name] = k

        return actions
