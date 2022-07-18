""" Define all the controllers availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from .base import Controller


class DecentralizedController(Controller):
    """DecentralizedController class to define a simple decentralized
    multi-agents setting.
    """

    def __init__(self, N, agent_cls, agent_kwargs):
        """Init."""
        self.done = False
        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs)

    @property
    def best_arms(self):
        """Return for each agent the estimated best arm. """
        best_arms = dict()
        for agent_name, agent in self.agents.items():
            best_arms[agent_name] = agent.best_arm
        return best_arms

    def act(self, observations, rewards):
        """Make each agent choose an arm in a decentralized way."""
        actions, dones = dict(), []
        for agent_name in observations.keys():

            observation = observations[agent_name]
            reward = rewards[agent_name]

            agent = self.agents[agent_name]

            k = agent.act(observation, reward)
            actions[agent_name] = k

            if hasattr(agent, 'done'):
                dones.append(agent.done)

        self.done = (len(dones) != 0) & all(dones)

        return actions
