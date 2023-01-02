""" Define all the controllers availables in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import itertools
import numpy as np
from scipy import linalg, optimize
import networkx as nx

from ._base import ControllerBase
from ..agents._linear_bandit_agents import LinUCB, LinUniform
from .._compils import _K_func
from .._checks import check_random_state, check_N_and_agent_names


class DecentralizedController(ControllerBase):
    """DecentralizedController class to define a simple decentralized
    multi-agents setting.
    """

    def __init__(self, agent_cls, agent_kwargs, N=None, agent_names=None,
                 seed=None):
        """Init."""
        self.done = False

        N, agent_names = check_N_and_agent_names(N, agent_names)

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def act(self, observations, rewards):
        """Make each agent choose an arm in a decentralized way."""
        actions, dones = dict(), []
        for agent_name in observations.keys():

            observation = observations[agent_name]
            reward = rewards[agent_name]

            agent = self.agents[agent_name]

            selected_k_or_arm = agent.act(observation, reward)
            actions[agent_name] = selected_k_or_arm

            if hasattr(agent, 'done'):
                dones.append(agent.done)

        self.done = (len(dones) != 0) & all(dones)

        return actions


class ClusteringController(ControllerBase):
    """ClusteringController class to define a simple clustered
    multi-agents.
    """

    def __init__(self, agent_cls, agent_kwargs, N=None, agent_names=None,
                 seed=None):
        """Init."""
        self.done = False

        N, agent_names = check_N_and_agent_names(N, agent_names)

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        raise NotImplementedError

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
        """Make each agent choose an arm in a clustered way."""

        agent_0 = list(observations.keys())[0]
        t = observations[agent_0]['t']  # fetch the iteration index

        if not self.labels_attributed:
            self.cluster_agents(t)

        # initialize the obersations/rewards per cluster
        observations_per_cluster, rewards_per_cluster = dict(), dict()
        for label in self.unique_labels:
            empty_observation = {'last_arm_pulled': [], 'last_reward': [],
                                 't': t}
            observations_per_cluster[label] = empty_observation
            rewards_per_cluster[label] = []

        # gather the observation for each cluster
        for agent_name in observations.keys():

            label = self.agent_labels[agent_name]

            k_or_arm = observations[agent_name]['last_arm_pulled']
            r = observations[agent_name]['last_reward']

            observations_per_cluster[label]['last_arm_pulled'].append(k_or_arm)
            observations_per_cluster[label]['last_reward'].append(r)

            rewards_per_cluster[label].append(r)

        # share the regrouped observations for each agent
        for agent_name in observations.keys():

            label = self.agent_labels[agent_name]

            selected_k_or_arms_shared = list(observations_per_cluster[label]['last_arm_pulled'])  # noqa
            selected_k_or_arms_local = observations[agent_name]['last_arm_pulled']  # noqa
            complete_selected_k_or_arms = (selected_k_or_arms_local, selected_k_or_arms_shared)  # noqa

            reward_shared = list(rewards_per_cluster[label])
            reward_local = float(observations[agent_name]['last_reward'])
            complete_reward = (reward_local, reward_shared)

            # overwrite the last observation/reward for each agent
            observations[agent_name]['last_arm_pulled'] = complete_selected_k_or_arms  # noqa
            rewards[agent_name] = complete_reward

        self.l_labels.append([self.agent_labels[agent_name]
                              for agent_name in self.agent_names])

        return self._act(observations, rewards)


class SingleClusterController(ClusteringController):
    """OracleClusteringController class to define a clustered
    multi-agents with true label at initialization.
    """

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
        self.unique_labels = np.array([0])
        self.l_labels = [[0] * N]

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        pass


class OracleClusteringController(ClusteringController):
    """OracleClusteringController class to define a clustered
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
        self.unique_labels = np.unique(list(self.agent_labels.values()))
        self.l_labels = [[self.agent_labels[agent_name]
                          for agent_name in agent_names]]

        super().__init__(N=N, agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        pass


class GraphController(ClusteringController):
    """GraphController class to define a simple clustered
    multi-agents setting.
    """

    def __init__(self, R, S, lbda, A_init, delta, agent_cls, agent_kwargs,
                 N=None, agent_names=None, seed=None):
        """Init."""
        self.done = False

        N, agent_names = check_N_and_agent_names(N, agent_names)

        # level-line of the ellipsoids
        self.delta = delta

        # upper-bound of the true theta norm
        self.S = S

        # R-sub-gaussianity of the noise (standard-deviation if Gaussian)
        self.R = R

        # ridge parameter
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
        self.unique_labels = np.arange(N)
        self.agent_labels = dict()
        for i, agent_name in enumerate(self.agent_names):
            self.agent_labels[agent_name] = i
        self.l_labels = [[self.agent_labels[agent_name]
                          for agent_name in self.agent_names]]

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

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""
        raise NotImplementedError


class KPartiteGraphController(GraphController):
    """Bi-partite GraphController. """
    def __init__(self, R, S, lbda, A_init, delta, n_clusters,
                 freq_graph_update, agent_cls, agent_kwargs, N=None,
                 agent_names=None, seed=None):
        """Init."""
        # adgency matrix of the graph
        self.freq_graph_update = freq_graph_update

        N, agent_names = check_N_and_agent_names(N, agent_names)

        # clustering parameters
        self.n_clusters = n_clusters

        super().__init__(N=N, R=R, S=S, lbda=lbda, A_init=A_init, delta=delta,
                         agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def _format_G(self, G, t):
        return (G < 0.0).astype(int)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""

        if t % self.freq_graph_update == 0:

            G = self.compute_graph()
            G = self._format_G(G, t)

            labels = nx.greedy_color(nx.from_numpy_array(G))
            labels = np.array([labels[i] for i in range(self.N)])
            self.unique_labels = np.unique(labels)
            n_comps = len(self.unique_labels)

            if n_comps == self.n_clusters:

                for i, label in enumerate(labels):
                    self.agent_labels[f"agent_{i}"] = label
                self.labels_attributed = True

            else:
                labels = np.arange(self.N)
                self.unique_labels = np.arange(self.N)
                self.labels_attributed = False


class LBC(GraphController):
    """Connected components GraphController. """
    def __init__(self, R, S, lbda, A_init, delta,
                 freq_graph_update, agent_cls, agent_kwargs, N=None,
                 agent_names=None, seed=None):
        """Init."""
        # adgency matrix of the graph
        self.freq_graph_update = freq_graph_update

        N, agent_names = check_N_and_agent_names(N, agent_names)

        super().__init__(N=N, R=R, S=S, lbda=lbda, A_init=A_init, delta=delta,
                         agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def _format_G(self, G, t):
        return (G >= 0).astype(int)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""

        if t % self.freq_graph_update == 0:

            # update graph
            G = self.compute_graph()

            # keep track of the graph
            self.G = G

            # preprocess the graph before clustering
            G = self._format_G(G, t)

            comps = list(nx.connected_components(nx.from_numpy_array(G)))
            n_comps = len(comps)

            for label, comp in enumerate(comps):
                for i in comp:
                    self.agent_labels[self.agent_names[i]] = label

            self.unique_labels = np.unique(range(n_comps))


class LBCSP(GraphController):
    """Connected components GraphController with two 'Separates Parts'. """
    def __init__(self, n_clusters, R, S, lbda, A_init, delta,
                 freq_graph_update, agent_cls, agent_kwargs, N=None,
                 agent_names=None, seed=None):
        """Init."""
        # adgency matrix of the graph
        self.freq_graph_update = freq_graph_update
        self.n_clusters = n_clusters

        N, agent_names = check_N_and_agent_names(N, agent_names)

        self.second_agent_cls = agent_cls
        self.second_agent_kwargs = agent_kwargs

        agent_cls = LinUniform
        agent_kwargs = {'arms': agent_kwargs.get('arms', None),
                        'arm_entries': agent_kwargs.get('arm_entries', None),
                        'te': agent_kwargs['te'],
                        'seed': agent_kwargs['seed'],
                        }

        super().__init__(N=N, R=R, S=S, lbda=lbda, A_init=A_init, delta=delta,
                         agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def _format_G(self, G, t):
        return (G >= 0).astype(int)

    def _switch_policy(self):
        """Switch the agent policy to a new one while keeping the gathered
        observation."""
        new_agents = dict()

        for i in range(self.N):

            agent = self.agents[f"agent_{i}"]
            new_agent = self.second_agent_cls(**self.second_agent_kwargs)
            new_agents[f"agent_{i}"] = new_agent

            new_agent.A = np.copy(agent.A)
            new_agent.b = np.copy(agent.b)
            new_agent.inv_A = np.copy(agent.inv_A)
            new_agent.theta_hat = np.copy(agent.theta_hat)
            new_agent.A_local = np.copy(agent.A_local)
            new_agent.b_local = np.copy(agent.b_local)
            new_agent.inv_A_local = np.copy(agent.inv_A_local)
            new_agent.theta_hat_local = np.copy(agent.theta_hat_local)
            new_agent.A_local = np.copy(agent.A_local)

        self.agents = new_agents

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""

        if t % self.freq_graph_update == 0:

            # update graph
            G = self.compute_graph()

            # keep track of the graph
            self.G = G

            # preprocess the graph before clustering
            G = self._format_G(G, t)

            comps = list(nx.connected_components(nx.from_numpy_array(G)))
            n_comps = len(comps)

            if self.n_clusters == n_comps:
                self._switch_policy()
                self.labels_attributed = True

            for label, comp in enumerate(comps):
                for i in comp:
                    self.agent_labels[self.agent_names[i]] = label

            self.unique_labels = np.unique(range(n_comps))


class LBCCP(GraphController):
    """Connected components GraphController with 'Coordonated Pullings'. """
    def __init__(self, R, S, lbda, A_init, delta,
                 freq_graph_update, agent_cls, agent_kwargs, N=None,
                 agent_names=None, seed=None):
        """Init."""
        # adgency matrix of the graph
        self.freq_graph_update = freq_graph_update

        N, agent_names = check_N_and_agent_names(N, agent_names)

        super().__init__(N=N, R=R, S=S, lbda=lbda, A_init=A_init, delta=delta,
                         agent_cls=agent_cls, agent_kwargs=agent_kwargs,
                         agent_names=agent_names, seed=seed)

    def _act(self, observations, rewards):
        """Make each agent choose an arm in a coordinate way."""
        # fetch the arms in a hackish way
        arms = self.agents['agent_0'].arms._arms

        actions, dones = dict(), []
        for i, agent_name in enumerate(observations.keys()):

            # update agent variables
            observation = observations[agent_name]
            reward = rewards[agent_name]

            agent = self.agents[agent_name]

            if self.G is None:
                k = agent.act(observation, reward)

            else:
                # proposed alternative arm pulling strategy
                idx, = np.where(self.G[i, :].ravel() > 0.0)
                j = self.rng.choice(idx)
                agent_j = self.agents[f"agent_{j}"]

                gap_ij = agent.theta_hat_local - agent_j.theta_hat_local

                aa = []
                for x_k in arms:
                    inv_A_x_k = np.linalg.pinv(agent.inv_A_local + x_k.T.dot(x_k))  # XXX # noqa
                    a = np.sqrt(gap_ij.T.dot(inv_A_x_k).dot(gap_ij))
                    aa.append(float(a))
                k = np.argmax(aa)

            # store action
            actions[agent_name] = k

            if hasattr(agent, 'done'):
                dones.append(agent.done)

        self.done = (len(dones) != 0) & all(dones)

        return actions

    def _format_G(self, G, t):
        return (G >= 0).astype(int)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""

        if t % self.freq_graph_update == 0:

            # update graph
            G = self.compute_graph()

            # keep track of the graph
            self.G = G

            # preprocess the graph before clustering
            G = self._format_G(G, t)

            comps = list(nx.connected_components(nx.from_numpy_array(G)))
            n_comps = len(comps)

            for label, comp in enumerate(comps):
                for i in comp:
                    self.agent_labels[self.agent_names[i]] = label

            self.unique_labels = np.unique(range(n_comps))


class CLUB(ClusteringController):
    """CLUB algorithm as defined in ```Online Clustering of
     Bandits```. """

    def __init__(self, gamma, agent_cls, agent_kwargs, N=None,
                 agent_names=None, seed=None):
        """Init."""
        self.done = False  # never stop clustering

        # clustering related savings
        self.gamma = gamma
        self.l_G = []
        self.l_comps = []
        self.l_n_comps = []

        super().__init__(N=N,
                         agent_cls=agent_cls,
                         agent_kwargs=agent_kwargs,
                         agent_names=agent_names,
                         seed=seed)

        # clusters variables
        self.labels_attributed = False
        self.unique_labels = np.arange(self.N)
        self.agent_labels = dict()
        for i in np.arange(self.N):
            self.agent_labels[f"agent_{i}"] = i
        self.l_labels = [[self.agent_labels[f"agent_{i}"]
                          for i in range(self.N)]]

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

    def _ub(self, t):
        return self.gamma * np.sqrt((1.0 + np.log(1.0 + t)) / (1.0 + t))

    def _format_G(self, G, t):
        return (G < self._ub(t)).astype(int)

    def cluster_agents(self, t):
        """Cluster all the agents from their estimated theta."""

        G = self.compute_graph()
        G = self._format_G(G, t)

        comps = list(nx.connected_components(nx.from_numpy_array(G)))
        n_comps = len(comps)

        for label, comp in enumerate(comps):
            for i in comp:
                self.agent_labels[self.agent_names[i]] = label

        self.unique_labels = np.unique(range(n_comps))


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
                                               te=10, seed=seed)

        # clusters variables
        self.n_clusters = n_clusters
        self.agent_labels = self._init_cluster_labels()
        self.agents_idx_per_cluters = self._update_agents_idx_per_cluters()
        cluster_thetas, cluster_inv_A = self._update_cluster_statistics()
        self.cluster_thetas = cluster_thetas
        self.cluster_inv_A = cluster_inv_A
        self.l_labels = [[self.agent_labels[f"agent_{i}"]
                          for i in range(self.N)]]

    def _init_cluster_labels(self):
        agent_labels = dict()
        for i in np.arange(self.N):
            agent_labels[f"agent_{i}"] = self.rng.randint(self.n_clusters)
        return agent_labels

    def _update_agents_idx_per_cluters(self):
        agents_idx_per_cluters = dict()
        for m in range(self.n_clusters):
            labels = [i for i in np.arange(self.N)
                      if self.agent_labels[f"agent_{i}"] == m]
            agents_idx_per_cluters[m] = labels
        return agents_idx_per_cluters

    def _update_cluster_statistics(self):
        cluster_thetas = dict()
        cluster_inv_A = dict()
        for m in range(self.n_clusters):
            A = np.zeros((self.d, self.d))
            b = np.zeros((self.d, 1))
            for i in self.agents_idx_per_cluters[m]:
                A += self.agents[f"agent_{i}"].A_local
                b += self.agents[f"agent_{i}"].b_local
            inv_A = np.linalg.pinv(A)
            cluster_thetas[m] = inv_A.dot(b)
            cluster_inv_A[m] = inv_A
        return cluster_thetas, cluster_inv_A

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

    def act(self, observations, rewards):
        """Make each agent choose an arm in a decentralized way."""
        actions, dones = dict(), []
        for agent_name in observations.keys():

            observation = observations[agent_name]
            reward = rewards[agent_name]

            agent = self.agents[agent_name]

            # fetch cluster variables
            agent_cluster_idx = self.agent_labels[agent_name]
            theta_cluster = self.cluster_thetas[agent_cluster_idx]
            inv_A_cluster = self.cluster_inv_A[agent_cluster_idx]

            t = observation['t']

            # pull an arm
            uu = []
            for x_k in self.arms:
                u = self.alpha * np.sqrt(x_k.T.dot(inv_A_cluster).dot(x_k))
                u *= np.sqrt(np.log(t + 1))
                u += theta_cluster.T.dot(x_k)
                uu.append(float(u))
            k = np.argmax(uu)

            actions[agent_name] = k

            # update local theta (ignore shared update)
            agent._update_all_statistics(observation, reward)

            # reassign agent cluster
            ll = []
            for m in range(self.n_clusters):
                theta_diff = agent.theta_hat_local - self.cluster_thetas[m]
                ll.append(np.linalg.norm(theta_diff))
            agent_new_cluster_idx = np.argmin(ll)
            self.agent_labels[agent_name] = agent_new_cluster_idx

            if hasattr(agent, 'done'):
                dones.append(agent.done)

        self.done = (len(dones) != 0) & all(dones)

        self.l_labels.append([self.agent_labels[f"agent_{i}"]
                              for i in range(self.N)])
        self.agents_idx_per_cluters = self._update_agents_idx_per_cluters()
        cluster_thetas, cluster_inv_A = self._update_cluster_statistics()
        self.cluster_thetas = cluster_thetas
        self.cluster_inv_A = cluster_inv_A

        return actions


class LOCB():
    """LOcal Clustering in Bandits (LOCB) as defined in ```Local Clustering in
    Contextual Multi-Armed Bandits```. """
    def __init__(self, N, alpha, delta, R, gamma, tau, n_clusters, arms, seed):
        """Init."""
        # ucb parameter
        self.alpha = alpha
        self.arms = arms
        self.d = len(self.arms[0])

        self.N = N

        self.done = False

        # random varaible
        self.rng = check_random_state(seed)

        # clustering parameters
        self.n_clusters = n_clusters
        self.l_labels = []
        self.users_seed = list(self.rng.choice(range(self.N), size=n_clusters))

        # current clusters
        self.clusters = dict()
        for user_seed in self.users_seed:
            self.clusters[user_seed] = [f'agent_{i}'
                                        for i in np.arange(self.N)]

        # cluster on which each agent appears
        self.agent_clusters = dict()
        for i in range(self.N):
            self.agent_clusters[f'agent_{i}'] = []

        self.delta = delta
        self.R = R
        self.gamma = gamma
        self.tau = tau
        self.H = self.delta / (2 * self.d)

        lbdas = np.linalg.eigvalsh(np.c_[self.arms].T.dot(np.c_[self.arms]))
        lbdas = np.unique(lbdas[lbdas > 0])
        self.lbda = np.min(lbdas)

        # init agents
        self.agents = dict()
        agent_kwargs = {'arms': arms, 'alpha': alpha, 'seed': seed}
        for n in range(self.N):
            self.agents[f"agent_{n}"] = LinUCB(**agent_kwargs)

    @property
    def best_arms(self):
        """Return for each agent the estimated best arm. """
        best_arms = dict()
        for agent_name, agent in self.agents.items():
            best_arms[agent_name] = agent.best_arm
        return best_arms

    def act(self, observations, rewards):
        """Make each agent choose an arm in a decentralized way."""

        agent_0 = list(observations.keys())[0]
        t = observations[agent_0]['t']
        alpha_ = self._alpha(t)

        # compute mean theta in each cluster and all the |x_k|_A for each agent
        theta_clusters, upper_bounds_clusters = dict(), dict()
        for user_seed in self.users_seed:

            upper_bounds_per_arms = dict()
            for k, x_k in enumerate(self.arms):

                all_cbs = []
                for agent_name_ in self.clusters[user_seed]:

                    inv_A_local = self.agents[agent_name_].inv_A_local
                    norm_ = x_k.T.dot(inv_A_local).dot(x_k)
                    all_cbs.append(alpha_ * np.sqrt(norm_))

                upper_bounds_per_arms[k] = np.mean(all_cbs)

            upper_bounds_clusters[user_seed] = upper_bounds_per_arms

            all_thetas = []
            for agent_name_ in self.clusters[user_seed]:

                theta = self.agents[agent_name_].theta_hat_local
                all_thetas.append(theta)

            theta_clusters[user_seed] = np.mean(all_thetas, axis=0)

        # arm pulling
        actions = dict()
        for agent_name in observations.keys():

            agent = self.agents[agent_name]

            # find agent currently belong to which clusters
            self.agent_clusters[agent_name] = []
            for user_seed in self.users_seed:
                if agent_name in self.clusters[user_seed]:
                    self.agent_clusters[agent_name].append(user_seed)

            # select an arm
            uu = []
            for k, x_k in enumerate(self.arms):

                vv = []
                for user_seed in self.agent_clusters[agent_name]:

                    r = float(theta_clusters[user_seed].T.dot(x_k))
                    ub = upper_bounds_clusters[user_seed][k]

                    vv.append(r + ub)
                uu.append(np.max(vv))

            actions[agent_name] = np.argmax(uu)

        # agent updates
        for agent_name in observations.keys():

            observation = observations[agent_name]
            reward = rewards[agent_name]

            # shared updates will not be used
            agent._update_all_statistics(observation, reward)

        # cluster agents
        if len(self.users_seed) == 0:
            self.done = True

        else:
            for agent_name in self.agents.keys():

                theta_agent = self.agents[agent_name].theta_hat_local

                for user_seed in self.users_seed:

                    theta_cluster = theta_clusters[user_seed]
                    norm_diff = np.linalg.norm(theta_agent - theta_cluster)

                    u = self._upper_bound(t)

                    # update cluster
                    if norm_diff > 2.0 * self._upper_bound(t):
                        self.clusters[user_seed].remove(agent_name)

                    # cluster converged
                    if u < self.gamma / 8.0 * self.tau:
                        self.users_seed.remove(user_seed)

        return actions

    def _upper_bound(self, t):
        """The cluster upper bound."""

        h = self.lbda * t / 4.0
        h -= 8.0 * np.log((t + 3.0) / self.H)
        h -= 2.0 * np.sqrt(t * np.log((t + 3.0) / self.H))

        bc = np.sqrt(2.0 * self.d * np.log(t) + 2.0 * np.log(2.0 / self.delta))
        bc = self.R * bc + 1

        if h > -1.0:  # at first iteration the bound's formula don't hold
            bc /= np.sqrt(1 + h)

        else:
            bc = np.inf

        return bc

    def _alpha(self, t):
        return np.sqrt(np.log(self.d * t))

    def init_act_randomly(self):
        """ Make each agent pulls randomly an arm to initiliaze the simulation.
        """
        actions = dict()
        for agent_name, agent in self.agents.items():
            actions[agent_name] = agent.randomly_select_arm()
        return actions

    def init_act(self, k=0):
        """ Make each agent pulls randomly an arm to initiliaze the simulation.
        """
        actions = dict()
        for agent_name, agent in self.agents.items():
            actions[agent_name] = k  # select the k-th arm for all agents
        return actions
