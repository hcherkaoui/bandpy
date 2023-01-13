""" Define all the controllers availables in Bandpy. """

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import itertools
import numpy as np
from scipy import linalg, optimize
import networkx as nx

from ._base import ControllerBase
from ..agents._linear_bandit_agents import LinUCB
from .._compils import _K_func
from .._checks import check_random_state, check_N_and_agent_names


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

        self.agents["agent_0"].update_local(last_k_or_arm, last_r, t)
        self.agents["agent_0"].update_shared(last_k_or_arm, last_r, t)

        action = {"agent_0": self.agents["agent_0"].act(t)}

        self.done = self.agents["agent_0"].done

        return action


class Decentralized(ControllerBase):
    """Decentralized class to define a simple decentralized
    multi-agents setting.
    """

    def __init__(self, agent_cls, agent_kwargs, N=None, agent_names=None,
                 seed=None):
        """Init."""
        self.done = False

        N, agent_names = check_N_and_agent_names(N, agent_names)

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def act(self, observation, reward):
        """Make a chosen agent choose an arm."""
        # check that the environment feedback concerns only the last agent
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg

        # fetch the agent name with the observation
        agent_name = next(iter(observation.keys()))

        # decompose the observation
        last_k_or_arm = observation[agent_name]['last_arm_pulled']
        last_r = observation[agent_name]['last_reward']
        t = observation[agent_name]['t']

        # update the agent with the observation
        self.agents[agent_name].update_local(last_k_or_arm, last_r, t)

        # selected one agent and make it pull an arm
        agent_name = self.choose_agent()
        selected_k_or_arm = self.agents[agent_name].act(t)
        action = {agent_name: selected_k_or_arm}

        # check if all agents are done
        dones = [agent.done for agent in self.agents.values()
                 if hasattr(agent, 'done')]
        self.done = (len(dones) != 0) & all(dones)

        return action


class ClusteringController(ControllerBase):
    """ClusteringController class to define a simple clustered
    multi-agents.
    """

    def __init__(self, agent_cls, agent_kwargs, N=None, agent_names=None,
                 seed=None):
        """Init."""
        self.done = False

        N, agent_names = check_N_and_agent_names(N, agent_names)

        # store the number of time an agent is chosen
        self.T_i = np.zeros((N,), dtype=int)

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        raise NotImplementedError

    def act(self, observation, reward):
        """Make each agent choose an arm in a clustered way."""
        # check that the environment feedback concerns only the last agent
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg

        # fetch the name of the agent with the observation
        agent_name_ref = next(iter(observation.keys()))

        # decompose the given observation
        last_k_or_arm = observation[agent_name_ref]['last_arm_pulled']
        last_r = observation[agent_name_ref]['last_reward']
        t = observation[agent_name_ref]['t']

        # update locally the agent
        self.agents[agent_name_ref].update_local(last_k_or_arm, last_r, t)

        # update the clustering structure
        if not self.labels_attributed:
            self.cluster_agents(t)

        # archive the clustering structure
        self.l_labels.append([self.agent_labels[agent_name]
                              for agent_name in self.agent_names])

        # fetch the agent label
        label_ref = self.agent_labels[agent_name_ref]

        # update all the agents of the cluster
        for agent_name, label in self.agent_labels.items():
            if label == label_ref:
                self.agents[agent_name].update_shared(last_k_or_arm, last_r, t)

        # selected one agent and make it pull an arm
        agent_name = self.choose_agent()
        selected_k_or_arm = self.agents[agent_name].act(t)
        action = {agent_name: selected_k_or_arm}

        # update the number of call for the chosen agent
        agent_idx = int(agent_name.split('_')[1])
        self.T_i[agent_idx] += 1

        # check if all agents are done
        dones = [agent.done for agent in self.agents.values()
                 if hasattr(agent, 'done')]
        self.done = (len(dones) != 0) & all(dones)

        return action


class SingleCluster(ClusteringController):
    """SingleCluster class to define a clustered multi-agents with a single
    cluster."""

    def __init__(self, agent_cls, agent_kwargs, N=None, agent_names=None,
                 seed=None):
        """Init."""
        self.done = False

        N, agent_names = check_N_and_agent_names(N, agent_names)

        # clusters variables
        self.labels_attributed = True

        self.agent_labels = dict()
        for agent_name in agent_names:
            self.agent_labels[agent_name] = 0

        self.l_labels = [[0] * N]

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        pass


class OracleClustering(ClusteringController):
    """OracleClustering class to define a clustered
    multi-agents with true label at initialization.
    """

    def __init__(self, agent_labels, agent_cls, agent_kwargs,
                 N=None, agent_names=None, seed=None):
        """Init."""
        self.done = False

        N, agent_names = check_N_and_agent_names(N, agent_names)

        # clusters variables
        self.labels_attributed = True

        self.agent_labels = agent_labels

        self.l_labels = [[self.agent_labels[agent_name]
                          for agent_name in agent_names]]

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        pass


class LBC(ClusteringController):
    """LBC algorithm. """

    def __init__(self,  R, S, lbda, A_init, delta, Tg, agent_cls, agent_kwargs,
                 N=None, agent_names=None, seed=None):
        """Init."""
        self.arms = agent_kwargs['arms']  # XXX
        self.alpha = agent_kwargs['alpha']  # XXX
        self.d = len(self.arms[0])  # XXX

        N, agent_names = check_N_and_agent_names(N, agent_names)

        self.done = False

        self.Tg = Tg
        self.delta = delta
        self.S = S
        self.R = R

        self.lbda = lbda
        self.A_init = A_init
        self.det_A_init = np.linalg.det(self.A_init)

        super().__init__(N=N,
                         agent_cls=agent_cls,
                         agent_kwargs=agent_kwargs,
                         agent_names=agent_names,
                         seed=seed)

        # graph
        self.G = None

        # clusters variables
        self.labels_attributed = False

        self.agent_labels = dict()
        for i in np.arange(self.N):
            self.agent_labels[f"agent_{i}"] = i

        self.comps = [set([i]) for i in range(N)]

        self.l_labels = [[self.agent_labels[f"agent_{i}"]
                          for i in range(self.N)]]

    def get_A_and_theta(self, cluster_idx):
        """Update A and b for cluster 'm'."""
        A = np.zeros((self.d, self.d))
        b = np.zeros((self.d, 1))
        for i in self.comps[cluster_idx]:
            A += self.agents[f"agent_{i}"].A_local
            b += self.agents[f"agent_{i}"].b_local
        inv_A = np.linalg.pinv(A)
        return inv_A, inv_A.dot(b)

    def act(self, observation, reward):
        """Make each agent choose an arm in a clustered way."""

        # check that the environment feedback concerns only the last agent
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg

        # fetch the name of the agent with the observation
        agent_name_ref = next(iter(observation.keys()))

        # decompose the given observation
        last_k_or_arm = observation[agent_name_ref]['last_arm_pulled']
        last_r = observation[agent_name_ref]['last_reward']
        t = observation[agent_name_ref]['t']

        # update locally the agent
        self.agents[agent_name_ref].update_local(last_k_or_arm, last_r, t)

        # update the clustering structure
        self.cluster_agents(t)

        # archive the clustering structure
        self.l_labels.append([self.agent_labels[agent_name]
                              for agent_name in self.agent_names])

        # selected one agent and make it pull an arm
        agent_name = self.choose_agent()
        cluster_idx = self.agent_labels[agent_name]
        inv_A_cluster, theta_cluster = self.get_A_and_theta(cluster_idx)

        # pull an arm
        uu = []
        for x_k in self.arms:
            u = self.alpha * np.sqrt(x_k.T.dot(inv_A_cluster).dot(x_k))
            u *= np.sqrt(np.log(t + 1))
            u += theta_cluster.T.dot(x_k)
            uu.append(float(u))
        k = np.argmax(uu)
        action = {agent_name: k}

        # update the number of call for the chosen agent
        agent_idx = int(agent_name.split('_')[1])
        self.T_i[agent_idx] += 1

        # check if all agents are done
        dones = [agent.done for agent in self.agents.values()
                 if hasattr(agent, 'done')]
        self.done = (len(dones) != 0) & all(dones)

        return action

    def min_K_func(self, inv_V_i, inv_V_j, theta_i, theta_j, eps_i, eps_j):
        """Test function to check if two ellipsoid are overlapping."""
        lambdas, phi = linalg.eigh(inv_V_j * eps_j, b=inv_V_i * eps_i)

        d, _ = theta_i.shape  # either theta_i or theta_j
        lambdas = lambdas.reshape((d, 1))
        phi = phi.reshape((d, d))

        theta_ij = theta_i - theta_j
        v_squared = phi.T.dot(theta_ij) ** 2

        def f(s):
            return float(_K_func(s, v_squared, lambdas))

        res = optimize.minimize_scalar(f, bounds=[0.0, 1.0], method='bounded')

        return float(res.fun)

    def eps(self, A):
        a = np.sqrt(self.lbda) * self.S
        b = 2.0 * np.log(1.0 / self.delta)
        c = np.log(np.linalg.det(A) / self.det_A_init)
        return a + self.R * np.sqrt(b + c)

    def compute_graph(self):
        """Compute the similarity graph G."""

        G = np.eye(self.N)

        for i, j in itertools.combinations(range(self.N), r=2):

            agent_i = self.agents[self.agent_names[i]]
            agent_j = self.agents[self.agent_names[j]]

            min_K = self.min_K_func(inv_V_i=agent_i.inv_A_local,
                                    inv_V_j=agent_j.inv_A_local,
                                    theta_i=agent_i.theta_hat_local,
                                    theta_j=agent_j.theta_hat_local,
                                    eps_i=self.eps(agent_i.A_local),
                                    eps_j=self.eps(agent_j.A_local))

            G[i, j] = min_K
            G[j, i] = min_K

        return G

    def preprocess_G(self, G, t):
        """Format the graph for the connected components estimation."""
        return (G >= 0).astype(int)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""

        if t % self.Tg == 0:

            G = self.compute_graph()
            self.G = G
            G = self.preprocess_G(G, t)

            self.comps = list(nx.connected_components(nx.from_numpy_array(G)))

            for label, comp in enumerate(self.comps):
                for i in comp:
                    self.agent_labels[self.agent_names[i]] = label


class LBCTwoPhases(ClusteringController):
    """LBC with two 'Separates Parts'. """

    def __init__(self,  n_clusters, R, S, lbda, A_init, delta,
                 Tg, explo_agent_cls, explo_agent_kwargs, agent_cls,
                 agent_kwargs, N=None, agent_names=None, seed=None):
        """Init."""
        self.arms = agent_kwargs['arms']  # XXX*
        self.K = len(self.arms)
        self.alpha = agent_kwargs['alpha']  # XXX
        self.d = len(self.arms[0])  # XXX

        N, agent_names = check_N_and_agent_names(N, agent_names)

        self.done = False  # never stop clustering

        self.Tg = Tg
        self.n_clusters = n_clusters
        self.delta = delta
        self.S = S
        self.R = R

        self.lbda = lbda
        self.A_init = A_init
        self.det_A_init = np.linalg.det(self.A_init)

        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs

        super().__init__(N=N,
                         agent_cls=self.agent_cls,
                         agent_kwargs=self.agent_kwargs,
                         agent_names=agent_names,
                         seed=seed)

        # graph
        self.G = None

        # clusters variables
        self.labels_attributed = False

        self.agent_labels = dict()
        for i in np.arange(self.N):
            self.agent_labels[f"agent_{i}"] = i

        self.comps = [set([i]) for i in range(N)]

        self.l_labels = [[self.agent_labels[f"agent_{i}"]
                          for i in range(self.N)]]

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""

        if t % self.Tg == 0:

            G = self.compute_graph()
            self.G = G
            G = self.preprocess_G(G, t)

            self.comps = list(nx.connected_components(nx.from_numpy_array(G)))
            n_comps = len(self.comps)

            if self.n_clusters == n_comps:
                self.labels_attributed = True

            for label, comp in enumerate(self.comps):
                for i in comp:
                    self.agent_labels[self.agent_names[i]] = label

    def get_A_and_theta(self, cluster_idx):
        """Update A and b for cluster 'm'."""
        A = np.zeros((self.d, self.d))
        b = np.zeros((self.d, 1))
        for i in self.comps[cluster_idx]:
            A += self.agents[f"agent_{i}"].A_local
            b += self.agents[f"agent_{i}"].b_local
        inv_A = np.linalg.pinv(A)
        return inv_A, inv_A.dot(b)

    def act(self, observation, reward):
        """Make each agent choose an arm in a clustered way."""

        # check that the environment feedback concerns only the last agent
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg

        # fetch the name of the agent with the observation
        agent_name_ref = next(iter(observation.keys()))

        # decompose the given observation
        last_k_or_arm = observation[agent_name_ref]['last_arm_pulled']
        last_r = observation[agent_name_ref]['last_reward']
        t = observation[agent_name_ref]['t']

        # update locally the agent
        self.agents[agent_name_ref].update_local(last_k_or_arm, last_r, t)

        # update the clustering structure
        if not self.labels_attributed:
            self.cluster_agents(t)

        # archive the clustering structure
        self.l_labels.append([self.agent_labels[agent_name]
                              for agent_name in self.agent_names])

        # selected one agent and make it pull an arm
        agent_name = self.choose_agent()

        if self.labels_attributed:
            cluster_idx = self.agent_labels[agent_name]
            inv_A_cluster, theta_cluster = self.get_A_and_theta(cluster_idx)

            # pull an arm
            uu = []
            for x_k in self.arms:
                u = self.alpha * np.sqrt(x_k.T.dot(inv_A_cluster).dot(x_k))
                u *= np.sqrt(np.log(t + 1))
                u += theta_cluster.T.dot(x_k)
                uu.append(float(u))
            k = np.argmax(uu)
            action = {agent_name: k}

        else:
            action = {agent_name: self.rng.randint(self.K)}

        # update the number of call for the chosen agent
        agent_idx = int(agent_name.split('_')[1])
        self.T_i[agent_idx] += 1

        # check if all agents are done
        dones = [agent.done for agent in self.agents.values()
                 if hasattr(agent, 'done')]
        self.done = (len(dones) != 0) & all(dones)

        return action

    def min_K_func(self, inv_V_i, inv_V_j, theta_i, theta_j, eps_i, eps_j):
        """Test function to check if two ellipsoid are overlapping."""
        lambdas, phi = linalg.eigh(inv_V_j * eps_j, b=inv_V_i * eps_i)

        d, _ = theta_i.shape  # either theta_i or theta_j
        lambdas = lambdas.reshape((d, 1))
        phi = phi.reshape((d, d))

        theta_ij = theta_i - theta_j
        v_squared = phi.T.dot(theta_ij) ** 2

        def f(s):
            return float(_K_func(s, v_squared, lambdas))

        res = optimize.minimize_scalar(f, bounds=[0.0, 1.0], method='bounded')

        return float(res.fun)

    def eps(self, A):
        a = np.sqrt(self.lbda) * self.S
        b = 2.0 * np.log(1.0 / self.delta)
        c = np.log(np.linalg.det(A) / self.det_A_init)
        return a + self.R * np.sqrt(b + c)

    def compute_graph(self):
        """Compute the similarity graph G."""

        G = np.eye(self.N)

        for i, j in itertools.combinations(range(self.N), r=2):

            agent_i = self.agents[self.agent_names[i]]
            agent_j = self.agents[self.agent_names[j]]

            min_K = self.min_K_func(inv_V_i=agent_i.inv_A_local,
                                    inv_V_j=agent_j.inv_A_local,
                                    theta_i=agent_i.theta_hat_local,
                                    theta_j=agent_j.theta_hat_local,
                                    eps_i=self.eps(agent_i.A_local),
                                    eps_j=self.eps(agent_j.A_local))

            G[i, j] = min_K
            G[j, i] = min_K

        return G

    def preprocess_G(self, G, t):
        """Format the graph for the connected components estimation."""
        return (G >= 0).astype(int)


class CLUB(ClusteringController):
    """CLUB algorithm as defined in ```Online Clustering of
     Bandits```. """

    def __init__(self, gamma, Tg, agent_cls, agent_kwargs, N=None,
                 agent_names=None, seed=None):
        """Init."""
        self.arms = agent_kwargs['arms']  # XXX
        self.alpha = agent_kwargs['alpha']  # XXX
        self.d = len(self.arms[0])  # XXX

        N, agent_names = check_N_and_agent_names(N, agent_names)

        self.done = False  # never stop clustering

        self.Tg = Tg

        # clustering related factor
        self.gamma = gamma

        super().__init__(N=N,
                         agent_cls=agent_cls,
                         agent_kwargs=agent_kwargs,
                         agent_names=agent_names,
                         seed=seed)

        # graph
        self.G = None

        # clusters variables
        self.labels_attributed = False

        self.agent_labels = dict()
        for i in np.arange(self.N):
            self.agent_labels[f"agent_{i}"] = 0

        self.comps = [set(range(N))]

        self.l_labels = [[self.agent_labels[f"agent_{i}"]
                          for i in range(self.N)]]

    def get_A_and_theta(self, cluster_idx):
        """Update A and b for cluster 'm'."""
        A = np.zeros((self.d, self.d))
        b = np.zeros((self.d, 1))
        for i in self.comps[cluster_idx]:
            A += self.agents[f"agent_{i}"].A_local
            b += self.agents[f"agent_{i}"].b_local
        inv_A = np.linalg.pinv(A)
        return inv_A, inv_A.dot(b)

    def act(self, observation, reward):
        """Make each agent choose an arm in a clustered way."""

        # check that the environment feedback concerns only the last agent
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg

        # fetch the name of the agent with the observation
        agent_name_ref = next(iter(observation.keys()))

        # decompose the given observation
        last_k_or_arm = observation[agent_name_ref]['last_arm_pulled']
        last_r = observation[agent_name_ref]['last_reward']
        t = observation[agent_name_ref]['t']

        # update locally the agent
        self.agents[agent_name_ref].update_local(last_k_or_arm, last_r, t)

        # update the clustering structure
        self.cluster_agents(t)

        # archive the clustering structure
        self.l_labels.append([self.agent_labels[agent_name]
                              for agent_name in self.agent_names])

        # selected one agent and make it pull an arm
        agent_name = self.choose_agent()
        cluster_idx = self.agent_labels[agent_name]
        inv_A_cluster, theta_cluster = self.get_A_and_theta(cluster_idx)

        # pull an arm
        uu = []
        for x_k in self.arms:
            u = self.alpha * np.sqrt(x_k.T.dot(inv_A_cluster).dot(x_k))
            u *= np.sqrt(np.log(t + 1))
            u += theta_cluster.T.dot(x_k)
            uu.append(float(u))
        k = np.argmax(uu)
        action = {agent_name: k}

        # update the number of call for the chosen agent
        agent_idx = int(agent_name.split('_')[1])
        self.T_i[agent_idx] += 1

        # check if all agents are done
        dones = [agent.done for agent in self.agents.values()
                 if hasattr(agent, 'done')]
        self.done = (len(dones) != 0) & all(dones)

        return action

    def compute_graph(self):
        """Compute the similarity graph G."""
        G = np.ones((self.N, self.N))

        for i, j in itertools.combinations(range(self.N), r=2):

            agent_i = self.agents[self.agent_names[i]]
            agent_j = self.agents[self.agent_names[j]]

            theta_i = agent_i.theta_hat_local
            theta_j = agent_j.theta_hat_local

            diff_thetas = np.linalg.norm(theta_i - theta_j)

            G[i, j] = diff_thetas
            G[j, i] = diff_thetas

        return G

    def compute_UB(self):
        """Compute the UB combination."""
        CB = []
        for i in range(self.N):
            cb = (1.0 + np.log(1.0 + self.T_i[i])) / (1.0 + self.T_i[i])
            CB.append(np.sqrt(cb))

        UB = np.ones((self.N, self.N))
        for i, j in itertools.combinations(range(self.N), r=2):
            UB[i, j] = self.gamma * (CB[i] + CB[j])
            UB[j, i] = self.gamma * (CB[i] + CB[j])

        return UB

    def preprocess_G(self, G, UB, t):
        """Format the graph for the connected components estimation."""
        return (G < UB).astype(int)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""

        if t % self.Tg == 0:

            UB = self.compute_UB()
            G = self.compute_graph()
            self.G = G
            G = self.preprocess_G(G, UB, t)

            self.comps = list(nx.connected_components(nx.from_numpy_array(G)))

            for label, comp in enumerate(self.comps):
                for i in comp:
                    self.agent_labels[self.agent_names[i]] = label


class DynUCB():
    """Dynamic UCB as defined in ```Dynamic Clustering of Contextual
    Multi-Armed Bandits```. """
    def __init__(self, N, alpha, n_clusters, arms, seed):
        """Init."""
        # ucb parameter
        self.alpha = alpha
        self.arms = arms
        self.d = len(self.arms[0])

        # clustering parameters
        self.n_clusters = n_clusters
        self.l_labels = []

        # random varaible
        self.rng = check_random_state(seed)

        # init agents
        self.N = N
        self.agents = dict()
        for n in range(self.N):
            # only use LinUCB for the internal variables (no arm selection)
            self.agents[f"agent_{n}"] = LinUCB(-1, arms=self.arms,
                                               arm_entries=None, lbda=1.0,
                                               seed=seed)

        # clusters variables
        self.n_clusters = n_clusters
        agent_labels, agents_idx_per_cluters = self._init_cluster_labels()
        self.agent_labels = agent_labels
        self.agents_idx_per_cluters = agents_idx_per_cluters

        self.cluster_thetas = dict()
        self.cluster_inv_A = dict()
        for cluster_idx in range(self.n_clusters):
            inv_A, theta = self.get_A_and_theta(cluster_idx)
            self.cluster_inv_A[cluster_idx] = inv_A
            self.cluster_thetas[cluster_idx] = theta

        self.l_labels = [[self.agent_labels[f"agent_{i}"]
                          for i in range(self.N)]]

    def _init_cluster_labels(self):
        """Init cluster labels."""
        agent_labels = dict()
        agents_idx_per_cluters = dict()

        for i in np.arange(self.N):
            agent_labels[f"agent_{i}"] = self.rng.randint(self.n_clusters)

        for m in range(self.n_clusters):
            labels = [i for i in np.arange(self.N)
                      if agent_labels[f"agent_{i}"] == m]
            agents_idx_per_cluters[m] = labels

        return agent_labels, agents_idx_per_cluters

    def get_A_and_theta(self, cluster_idx):
        """Update A and b for cluster 'm'."""
        A = np.zeros((self.d, self.d))
        b = np.zeros((self.d, 1))
        for i in self.agents_idx_per_cluters[cluster_idx]:
            A += self.agents[f"agent_{i}"].A_local
            b += self.agents[f"agent_{i}"].b_local
        inv_A = np.linalg.pinv(A)
        return inv_A, inv_A.dot(b)

    @property
    def best_arms(self):
        """Return for each agent the estimated best arm. """
        best_arms = dict()
        for agent_name, agent in self.agents.items():
            best_arms[agent_name] = agent.best_arm
        return best_arms

    def default_act(self):
        """ Make each agent pulls 'default' arm to init the simulation."""
        actions = dict()
        for agent_name, agent in self.agents.items():
            actions[agent_name] = self.agents[agent_name].select_default_arm()
        return actions

    def choose_agent(self):
        """Randomly return the name of an agent."""
        return f"agent_{self.rng.randint(self.N)}"

    def act(self, observation, reward):
        """Make each agent choose an arm in a decentralized way."""
        # check that the environment feedback concerns only the last agent
        msg = "'.act()' should manage only one agent at a time"
        assert len(observation) == 1, msg
        assert len(reward) == 1, msg

        # fetch the name of the agent with the observation
        last_agent_name = next(iter(observation.keys()))

        # decompose the given observation
        last_k_or_arm = observation[last_agent_name]['last_arm_pulled']
        last_r = observation[last_agent_name]['last_reward']
        t = observation[last_agent_name]['t']

        # update the last agent
        self.agents[last_agent_name].update_local(last_k_or_arm, last_r, t)

        # reassign the chosen agent to a new cluster
        ll = []
        old_agent_label = self.agent_labels[last_agent_name]
        for m in range(self.n_clusters):
            theta_local = self.agents[last_agent_name].theta_hat_local
            theta_cluster = self.cluster_thetas[m]
            ll.append(np.linalg.norm(theta_local - theta_cluster))
        agent_label = np.argmin(ll)
        self.agent_labels[last_agent_name] = agent_label

        # update all cluster variables related
        if agent_label != old_agent_label:
            agent_idx = int(last_agent_name.split('_')[1])

            # update old agent cluster
            self.agents_idx_per_cluters[old_agent_label].remove(agent_idx)
            inv_A_cluster, theta_cluster = self.get_A_and_theta(old_agent_label)  # noqa
            self.cluster_thetas[old_agent_label] = theta_cluster
            self.cluster_inv_A[old_agent_label] = inv_A_cluster

            # update new agent cluster
            self.agents_idx_per_cluters[agent_label].append(agent_idx)
            inv_A_cluster, theta_cluster = self.get_A_and_theta(agent_label)
            self.cluster_thetas[agent_label] = theta_cluster
            self.cluster_inv_A[agent_label] = inv_A_cluster

        # selected one agent
        agent_name = self.choose_agent()

        # take the chosen agent cluster variables
        theta_cluster = self.cluster_thetas[self.agent_labels[agent_name]]
        inv_A_cluster = self.cluster_inv_A[self.agent_labels[agent_name]]

        # pull an arm
        uu = []
        for x_k in self.arms:
            u = self.alpha * np.sqrt(x_k.T.dot(inv_A_cluster).dot(x_k))
            u *= np.sqrt(np.log(t + 1))
            u += theta_cluster.T.dot(x_k)
            uu.append(float(u))
        k = np.argmax(uu)
        action = {agent_name: k}

        # archive agent labels
        l_labels = [self.agent_labels[f"agent_{i}"] for i in range(self.N)]
        self.l_labels.append(l_labels)

        # check if all agents are done
        dones = [agent.done for agent in self.agents.values()
                 if hasattr(agent, 'done')]
        self.done = (len(dones) != 0) & all(dones)

        return action
