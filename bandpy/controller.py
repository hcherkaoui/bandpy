""" Define all the controllers availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
from sklearn.cluster import KMeans
from .base import Controller
from .agents import GreedyLinGapE


class DecentralizedController(Controller):
    """DecentralizedController class to define a simple decentralized
    multi-agents setting.
    """

    def __init__(self, N, agent_cls, agent_kwargs):
        """Init."""
        self.done = False
        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs)

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


class ClusteredLinearAgentsController(Controller):
    """ClusteredAgentsController class to define a simple decentralized
    multi-agents setting where linear bandit agents are clustered.
    """

    def __init__(self, N, n_clusters, m, agent_kwargs):
        """Init."""
        self.done = False

        # number of exploration rounds to estimate the clusters
        self.m = m

        # clusters variables
        self.labels_attributed = False
        self.n_clusters = n_clusters
        self.agent_labels = dict()
        self.unique_labels = []

        super().__init__(N=N, agent_cls=GreedyLinGapE,
                         agent_kwargs=agent_kwargs)

    def cluster_agents(self):
        """Cluster all the agents from their estimated theta."""
        X = np.array([agent.theta_hat.flatten()
                      for agent in self.agents.values()])

        kmeans = KMeans(n_clusters=self.n_clusters).fit(X)
        self.raw_est_thetas = [raw_est_thetas.flatten()
                               for raw_est_thetas in kmeans.cluster_centers_]

        unique_labels = []
        for agent_name, label in zip(self.agents.keys(), kmeans.labels_):
            self.agent_labels[agent_name] = label
            unique_labels.append(label)
        self.unique_labels = np.unique(unique_labels)

    def _act(self, observations, rewards):
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

    def act(self, observations, rewards):
        """Make each agent choose an arm in a decentralized way."""

        t = observations['agent_0']['t']  # fetch the iteration index

        if t > self.m:

            # when all the agents try all arm once, cluster them
            if not self.labels_attributed:
                self.cluster_agents()
                self.labels_attributed = True

            # if the clustered labels are available: add the observations
            # among same cluster

            # initialize the obersations/rewards per cluster
            observations_per_cluster, rewards_per_cluster = dict(), dict()
            for label in self.unique_labels:
                empty_observation = {'last_arm_pulled': [], 'last_reward': [],
                                     't': t,}
                observations_per_cluster[label] = empty_observation
                rewards_per_cluster[label] = []

            # gather the observation for each cluster
            for agent_name in observations.keys():

                label = self.agent_labels[agent_name]

                k = observations[agent_name]['last_arm_pulled']
                r = observations[agent_name]['last_reward']

                observations_per_cluster[label]['last_arm_pulled'].append(k)
                observations_per_cluster[label]['last_reward'].append(r)

                rewards_per_cluster[label].append(r)

            # share the regrouped observations for each agent
            for agent_name in observations.keys():

                label = self.agent_labels[agent_name]

                observations[agent_name] = observations_per_cluster[label]
                rewards[agent_name] = rewards_per_cluster[label]

        return self._act(observations, rewards)
