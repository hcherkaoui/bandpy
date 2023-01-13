""" Define all the agents availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from ..utils import check_random_state


class ControllerBase:
    """ Abstract class for a controller then handle multiple agents.

    Parameters
    ----------
    """

    def __init__(self, N, agent_cls, agent_kwargs, agent_names=None,
                 seed=None):
        """Init."""
        self.agents = dict()

        self.N = N

        self.seed = seed
        self.rng = check_random_state(seed)

        if agent_names is None:
            self.agent_names = [f"agent_{i}" for i in range(self.N)]
        else:
            self.agent_names = agent_names

        for agent_name in self.agent_names:
            self.agents[agent_name] = agent_cls(**agent_kwargs)

    def choose_agent(self):
        """Randomly return the name of an agent."""
        return f"agent_{self.rng.randint(self.N)}"

    @property
    def best_arms(self):
        """Return for each agent the estimated best arm.

        Parameters
        ----------
        """
        best_arms = dict()
        for agent_name, agent in self.agents.items():
            best_arms[agent_name] = agent.best_arm
        return best_arms

    def default_act(self):
        """ Make each agent pulls 'default' arm to init the simulation.

        Parameters
        ----------
        """
        agent_name = self.choose_agent()
        agent = self.agents[agent_name]
        return {agent_name: agent.select_default_arm()}
