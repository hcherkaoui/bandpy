""" Define all the controllers availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import numpy as np
import networkx as nx

from ._base import ControllerBase
from ..agents._linear_bandit_agents import MultiLinearAgentsBase
from .._compils import K_min
from .._checks import check_random_state, check_N_and_agent_names, check_A_init
from .._criterions import _f_ucb


class SingleAgentController:
    """Proxy for single agent case."""

    def __init__(self, agent_instance):
        """Init."""
        self.agents = {"agent_0": agent_instance}

    def choose_agent(self):
        return "agent_0"

    @property
    def best_arms(self):
        return {"agent_0": self.agents["agent_0"].best_arm}

    def default_act(self):
        return {"agent_0": self.agents["agent_0"].select_default_arm()}

    def act(self, observation, reward):
        """Make a chosen agent choose an arm."""
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg

        last_k_or_arm = observation["agent_0"]['last_arm_pulled']
        last_r = observation["agent_0"]['last_reward']
        t = observation["agent_0"]['t']

        self.agents["agent_0"].update_local(last_k_or_arm, last_r)
        self.agents["agent_0"].update_shared(last_k_or_arm, last_r)

        action = {"agent_0": self.agents["agent_0"].act(t)}

        self.done = self.agents["agent_0"].done

        return action


class ClusteringController(ControllerBase):
    """ClusteringController class to define a simple clustered
    multi-agents.
    """
    def archive_labels(self):
        """Archive the current agent labels within self.l_labels."""
        self.l_labels.append([self.agent_labels[agent_name]
                              for agent_name in self.agent_names])

    def check_if_controller_done(self):
        """Check if the controller return 'done'."""
        dones = [agent.done for agent in self.agents.values()
                 if hasattr(agent, 'done')]
        self.done = (len(dones) != 0) & all(dones)

    def check_observation_reward(self, observation, reward):
        """check that the environment feedback concerns only the last agent."""
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg
        return observation, reward

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        raise NotImplementedError

    def act(self, observation, reward):
        """Make each agent choose an arm in a clustered way."""
        self.check_observation_reward(observation, reward)

        last_agent_name = next(iter(observation.keys()))

        last_k_or_arm = observation[last_agent_name]['last_arm_pulled']
        last_r = observation[last_agent_name]['last_reward']
        t = observation[last_agent_name]['t']

        self.agents[last_agent_name].update_local(last_k_or_arm, last_r)

        self.cluster_agents(t)

        self.archive_labels()

        last_agent_new_label = self.agent_labels[last_agent_name]

        for agent_name, label in self.agent_labels.items():
            if label == last_agent_new_label:
                self.agents[agent_name].update_shared(last_k_or_arm, last_r)

        agent_name = self.choose_agent()
        selected_k_or_arm = self.agents[agent_name].act(t)

        self.check_if_controller_done()

        return {agent_name: selected_k_or_arm}


class SingleCluster(ClusteringController):
    """SingleCluster class to define a clustered multi-agents with a single
    cluster."""

    def __init__(self, agent_cls, agent_kwargs, N=None, agent_names=None,
                 seed=None):
        """Init."""
        N, agent_names = check_N_and_agent_names(N, agent_names)
        self.agent_labels = dict(zip([f"agent_{i}" for i in range(N)],
                                     [0] * N))
        self.l_labels = [list(np.zeros(N))]

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        pass


class Decentralized(ClusteringController):
    """Decentralized class to define a clustered multi-agents with a no
    cluster."""

    def __init__(self, agent_cls, agent_kwargs, N=None, agent_names=None,
                 seed=None):
        """Init."""
        N, agent_names = check_N_and_agent_names(N, agent_names)
        self.agent_labels = dict(zip([f"agent_{i}" for i in range(N)],
                                     list(range(N))))
        self.l_labels = [list(np.arange(N))]

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        pass


class OracleClustering(ClusteringController):
    """OracleClustering class to define a clustered multi-agents with true
    label."""

    def __init__(self, agent_labels, agent_cls, agent_kwargs,
                 N=None, agent_names=None, seed=None):
        """Init."""
        N, agent_names = check_N_and_agent_names(N, agent_names)
        self.agent_labels = agent_labels
        self.l_labels = [[self.agent_labels[agent_name]
                          for agent_name in agent_names]]

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        pass


class AbstractCLUB(ClusteringController):
    """AbstractCLUB class to define a general CLUB controller to derive the
    other graph-like controllers."""

    def pull_arm_ucb(self, t, inv_A, theta):
        """Return the best arm to pull following the UCB criterion."""
        uu = []
        for x_k in self.arms:
            uu.append(_f_ucb(self.alpha, t, x_k.ravel(), theta.ravel(), inv_A))
        return np.argmax(uu)

    def get_cluster_shared_parameters(self, cluster_idx):
        """Get the parameters of 'cluster_idx'."""
        A = np.copy(self.A_init)
        b = np.zeros((self.d, 1))
        for i in self.comps[cluster_idx]:
            A += self.agents[f"agent_{i}"].A_local - self.A_init
            b += self.agents[f"agent_{i}"].b_local
        inv_A = np.linalg.inv(A)
        theta = inv_A.dot(b)
        return A, b, inv_A, theta

    def update_shared_parameters(self, cluster_idx, A, b, inv_A, theta):
        """Update parameter for cluster 'cluster_idx'."""
        for i in self.comps[cluster_idx]:
            self.agents[f"agent_{i}"].A = np.copy(A)
            self.agents[f"agent_{i}"].b = np.copy(b)
            self.agents[f"agent_{i}"].inv_A = np.copy(inv_A)
            self.agents[f"agent_{i}"].chol_A = None
            self.agents[f"agent_{i}"].det_A = None
            self.agents[f"agent_{i}"].theta_hat = np.copy(theta)

    def act(self, observation, reward):
        """Make each agent choose an arm in a clustered way."""
        observation, reward = self.check_observation_reward(observation,
                                                            reward)

        last_agent_name = next(iter(observation.keys()))
        last_agent_i = int(last_agent_name.split('_')[1])

        last_k_or_arm = observation[last_agent_name]['last_arm_pulled']
        last_r = observation[last_agent_name]['last_reward']
        t = observation[last_agent_name]['t']

        self.agents[last_agent_name]._update_local(last_k_or_arm, last_r)

        self.update_clusters(last_agent_i)
        self.archive_labels()

        agent_name = self.choose_agent()

        cluster_idx = self.agent_labels[agent_name]
        A, b, inv_A, theta = self.get_cluster_shared_parameters(cluster_idx)
        self.update_shared_parameters(cluster_idx, A, b, inv_A, theta)
        k = self.pull_arm_ucb(t, inv_A, theta)

        self.check_if_controller_done()

        return {agent_name: k}


class CLUB(AbstractCLUB):
    """CLUB algorithm as defined in ```Online Clustering of
     Bandits```. """

    def __init__(self,
                 arms,
                 A_init=None,
                 gamma=1.0,
                 alpha=1.0,
                 lbda=1.0,
                 N=None,
                 agent_names=None,
                 seed=None):
        """Init."""
        # general parameters
        self.arms = arms
        self.alpha = alpha
        self.d = len(self.arms[0])
        self.A_init = check_A_init(self.d, lbda, A_init)
        N, agent_names = check_N_and_agent_names(N, agent_names)

        # CLUB related parameters
        self.gamma = gamma

        # super init
        agent_kwargs = dict(arms=arms,
                            A_init=self.A_init,
                            lbda=lbda,
                            seed=seed)
        super().__init__(N=N,
                         agent_cls=MultiLinearAgentsBase,
                         agent_kwargs=agent_kwargs,
                         agent_names=agent_names,
                         seed=seed)

        # graph related parameters
        self.graph_G = nx.from_numpy_array(np.ones((self.N, self.N)))
        self.agent_labels = dict(zip([f"agent_{i}" for i in range(N)],
                                     [0] * N))
        self.comps = [set(range(N))]
        self.l_labels = [list(np.zeros(N))]

    def update_clusters(self, i):
        """Update raw/col i-th of the similarity graph G."""
        cb_i = (1.0 + np.log(1.0 + self.T_i[i])) / (1.0 + self.T_i[i])
        theta_i = self.agents[self.agent_names[i]].theta_hat_local

        for j in range(self.N):

            cb_j = (1.0 + np.log(1.0 + self.T_i[j])) / (1.0 + self.T_i[j])
            ub = self.gamma * (cb_i + cb_j)

            theta_j = self.agents[self.agent_names[j]].theta_hat_local
            diff_thetas = (theta_i - theta_j).ravel()
            norm_diff_thetas = np.sqrt(diff_thetas.dot(diff_thetas))

            if norm_diff_thetas > ub:
                if self.graph_G.has_edge(i, j):
                    self.graph_G.remove_edge(i, j)

        self.comps = list(nx.connected_components(self.graph_G))

        for label, comp in enumerate(self.comps):
            for i in comp:
                self.agent_labels[self.agent_names[i]] = label


class LBC(AbstractCLUB):
    """LBC algorithm. """

    def __init__(self,
                 arms,
                 A_init=None,
                 R=1.0,
                 S=1.0,
                 delta=1e-3,
                 alpha=1.0,
                 lbda=1.0,
                 N=None,
                 agent_names=None,
                 seed=None):
        """Init."""
        # general parameters
        self.arms = arms
        self.alpha = alpha
        self.d = len(self.arms[0])
        self.A_init = check_A_init(self.d, lbda, A_init)
        N, agent_names = check_N_and_agent_names(N, agent_names)

        # LBC related parameters
        self.delta = delta
        self.S = S
        self.R = R
        self.lbda = lbda

        self.det_A_init = np.linalg.det(self.A_init)
        self.a = np.sqrt(self.lbda) * self.S
        self.b = 2.0 * np.log(1.0 / self.delta)

        # super init
        agent_kwargs = dict(arms=arms,
                            A_init=self.A_init,
                            lbda=lbda,
                            seed=seed)
        super().__init__(N=N,
                         agent_cls=MultiLinearAgentsBase,
                         agent_kwargs=agent_kwargs,
                         agent_names=agent_names,
                         seed=seed)

        # graph related parameters
        self.graph_G = nx.from_numpy_array(np.ones((self.N, self.N)))
        self.agent_labels = dict(zip([f"agent_{i}" for i in range(N)],
                                     [0] * N))
        self.comps = [set(range(N))]
        self.l_labels = [list(np.zeros(N))]

    def update_clusters(self, i):
        """Update raw/col i-th of the similarity graph G."""
        agent_i = self.agents[self.agent_names[i]]
        eps_i = self.a + self.R * np.sqrt(self.b + np.log(agent_i.det_A_local / self.det_A_init))  # noqa
        inv_A_i = agent_i.inv_A_local * eps_i
        theta_i = agent_i.theta_hat_local

        for j in set(self.graph_G.neighbors(i)) - set([i]):

            agent_j = self.agents[self.agent_names[j]]
            eps_j = self.a + self.R * np.sqrt(self.b + np.log(agent_j.det_A_local / self.det_A_init))  # noqa
            cho_A_j = agent_j.chol_A_local / np.sqrt(eps_j)
            theta_j = agent_j.theta_hat_local

            _, min_K = K_min(inv_A_i=inv_A_i, cho_A_j=cho_A_j, theta_i=theta_i,
                             theta_j=theta_j)

            if min_K < 0.0:
                if self.graph_G.has_edge(i, j):
                    self.graph_G.remove_edge(i, j)

        self.comps = list(nx.connected_components(self.graph_G))

        for label, comp in enumerate(self.comps):
            for i in comp:
                self.agent_labels[self.agent_names[i]] = label


class DynUCB():
    """Dynamic UCB as defined in ```Dynamic Clustering of Contextual
    Multi-Armed Bandits```. """
    def __init__(self, N, alpha, n_clusters, arms, A_init=None, lbda=1.0,
                 seed=None):
        """Init."""
        # ucb parameter
        self.alpha = alpha
        self.arms = arms
        self.d = len(self.arms[0])

        # clustering parameters
        self.n_clusters = n_clusters
        self.l_labels = []

        self.A_init = check_A_init(self.d, lbda, A_init)

        # random varaible
        self.rng = check_random_state(seed)

        agent_kwargs = dict(arms=self.arms,
                            A_init=self.A_init,
                            lbda=lbda,
                            seed=seed)

        # init agents
        self.N = N
        self.T_i = np.zeros((N,), dtype=int)
        self.t = -1  # to be synchonous with the env clock
        self.agents = dict()
        for n in range(self.N):
            self.agents[f"agent_{n}"] = MultiLinearAgentsBase(**agent_kwargs)

        # clusters variables
        self.n_clusters = n_clusters
        agent_labels, comps = self._init_cluster_labels()
        self.agent_labels = agent_labels
        self.comps = comps

        self.cluster_thetas = dict()
        self.cluster_inv_A = dict()
        for cluster_idx in range(self.n_clusters):
            _, _, inv_A, theta = self.get_cluster_shared_parameters(cluster_idx)  # noqa
            self.cluster_inv_A[cluster_idx] = inv_A
            self.cluster_thetas[cluster_idx] = theta

        self.l_labels = [[self.agent_labels[f"agent_{i}"]
                          for i in range(self.N)]]

        self.done = False

    def _init_cluster_labels(self):
        """Init cluster labels."""
        agent_labels = dict()
        for i in np.arange(self.N):
            agent_labels[f"agent_{i}"] = self.rng.randint(self.n_clusters)
        comps = dict()
        for m in range(self.n_clusters):
            labels = [i for i in np.arange(self.N)
                      if agent_labels[f"agent_{i}"] == m]
            comps[m] = labels
        return agent_labels, comps

    def get_cluster_shared_parameters(self, cluster_idx):
        """Get the parameters of 'cluster_idx'."""
        A = np.copy(self.A_init)
        b = np.zeros((self.d, 1))
        for i in self.comps[cluster_idx]:
            A += self.agents[f"agent_{i}"].A_local - self.A_init
            b += self.agents[f"agent_{i}"].b_local
        inv_A = np.linalg.inv(A)
        theta = inv_A.dot(b)
        return A, b, inv_A, theta

    def update_shared_parameters(self, cluster_idx, A, b, inv_A, theta):
        """Update parameter for cluster 'cluster_idx'."""
        for i in self.comps[cluster_idx]:
            self.agents[f"agent_{i}"].A = np.copy(A)
            self.agents[f"agent_{i}"].b = np.copy(b)
            self.agents[f"agent_{i}"].inv_A = np.copy(inv_A)
            self.agents[f"agent_{i}"].chol_A = None
            self.agents[f"agent_{i}"].det_A = None
            self.agents[f"agent_{i}"].theta_hat = np.copy(theta)

    @property
    def best_arms(self):
        """Return for each agent the estimated best arm. """
        best_arms = dict()
        for agent_name, agent in self.agents.items():
            best_arms[agent_name] = agent.best_arm
        return best_arms

    def default_act(self):
        """ Choose one agent and makes it pull the 'default' arm to init the
        simulation. """
        agent_name = self.choose_agent()
        agent = self.agents[agent_name]
        return {agent_name: agent.select_default_arm()}

    def choose_agent(self):
        """Randomly return the name of an agent."""
        i = self.rng.randint(self.N)
        self.t += 1  # asynchrone case
        self.T_i[i] += 1
        return f"agent_{i}"

    def pull_arm_ucb(self, t, inv_A, theta):
        """Return the best arm to pull following the UCB criterion."""
        uu = []
        for x_k in self.arms:
            uu.append(_f_ucb(self.alpha, t, x_k.ravel(), theta.ravel(), inv_A))
        return np.argmax(uu)

    def archive_labels(self):
        """Archive the current agent labels within self.l_labels."""
        l_labels = [self.agent_labels[f"agent_{i}"] for i in range(self.N)]
        self.l_labels.append(l_labels)

    def check_if_controller_done(self):
        """Check if the controller return 'done'."""
        dones = [agent.done for agent in self.agents.values()
                 if hasattr(agent, 'done')]
        self.done = (len(dones) != 0) & all(dones)

    def check_observation_reward(self, observation, reward):
        """check that the environment feedback concerns only the last agent."""
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg
        return observation, reward

    def act(self, observation, reward):
        """Make each agent choose an arm in a decentralized way."""
        observation, reward = self.check_observation_reward(observation,
                                                            reward)

        # fetch the name of the agent with the observation
        last_agent_name = next(iter(observation.keys()))

        last_k_or_arm = observation[last_agent_name]['last_arm_pulled']
        last_r = observation[last_agent_name]['last_reward']
        t = observation[last_agent_name]['t']

        # update the last agent
        self.agents[last_agent_name]._update_local(last_k_or_arm, last_r)

        # reassign the chosen agent to a new cluster
        ll = []
        old_agent_label = self.agent_labels[last_agent_name]
        for m in range(self.n_clusters):
            theta_local = self.agents[last_agent_name].theta_hat_local
            theta_cluster = self.cluster_thetas[m]
            diff_theta = (theta_local - theta_cluster).ravel()
            norm_diff_theta = np.sqrt(diff_theta.dot(diff_theta))
            ll.append(norm_diff_theta)
        agent_label = np.argmin(ll)
        self.agent_labels[last_agent_name] = agent_label

        # update all cluster variables related
        if agent_label != old_agent_label:
            agent_idx = int(last_agent_name.split('_')[1])

            # update old agent cluster
            self.comps[old_agent_label].remove(agent_idx)
            A_cluster, b_cluster, inv_A_cluster, theta_cluster = self.get_cluster_shared_parameters(old_agent_label)  # noqa
            self.update_shared_parameters(old_agent_label, A_cluster, b_cluster, inv_A_cluster, theta_cluster)  # noqa
            self.cluster_thetas[old_agent_label] = theta_cluster
            self.cluster_inv_A[old_agent_label] = inv_A_cluster

            # update new agent cluster
            self.comps[agent_label].append(agent_idx)
            A_cluster, b_cluster, inv_A_cluster, theta_cluster = self.get_cluster_shared_parameters(agent_label)  # noqa
            self.update_shared_parameters(agent_label, A_cluster, b_cluster, inv_A_cluster, theta_cluster)  # noqa

            self.cluster_thetas[agent_label] = theta_cluster
            self.cluster_inv_A[agent_label] = inv_A_cluster

        agent_name = self.choose_agent()

        # take the chosen agent cluster variables
        theta_cluster = self.cluster_thetas[self.agent_labels[agent_name]]
        inv_A_cluster = self.cluster_inv_A[self.agent_labels[agent_name]]

        k = self.pull_arm_ucb(t, inv_A_cluster, theta_cluster)

        self.archive_labels()

        self.check_if_controller_done()

        return {agent_name: k}
