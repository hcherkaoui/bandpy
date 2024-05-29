""" Define all the agents availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from .._checks import check_random_state


class ControllerBase:
    """Abstract class for a controller then handle multiple agents.

    Parameters
    ----------
    """

    def __init__(
        self,
        N,
        agent_cls,
        agent_kwargs,
        agent_names=None,
        agent_selection_type="random",
        seed=None,
    ):
        """Init."""
        self.agents = dict()

        self.N = N

        self.t = -1  # to be synchonous with the env count
        self.T_i = np.zeros((N,), dtype=int)

        self.seed = seed
        self.rng = check_random_state(seed)

        if agent_names is None:
            self.agent_names = [f"agent_{i}" for i in range(self.N)]
        else:
            self.agent_names = agent_names

        for agent_name in self.agent_names:
            self.agents[agent_name] = agent_cls(**agent_kwargs)

        self.agent_selection_type = agent_selection_type

        self.done = False

    def reset(self, seed=None):
        """Reset internal statistics."""
        self.seed = seed
        self.rng = check_random_state(self.seed)
        # self.init_metrics()
        self.t = -1

    def _choose_agent(self):
        """Randomly return the name of an agent."""
        if self.agent_selection_type == "random":
            i = self.rng.integers(self.N)

        elif self.agent_selection_type == "iterative":
            i = self.t % self.N

        else:
            raise ValueError(
                "Agent selection type not understood, got {self.agent_selection_type}"
            )

        self.t += 1  # asynchrone case
        self.T_i[i] += 1

        return f"agent_{i}"

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
        """Choose one agent and makes it pull the 'default' arm to init the
        simulation."""
        agent_name = self._choose_agent()
        agent = self.agents[agent_name]
        return {agent_name: agent.select_default_arm()}
