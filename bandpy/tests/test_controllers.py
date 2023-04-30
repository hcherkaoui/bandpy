"""Testing module for the compils functions. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import pytest
import numpy as np
from sklearn import metrics
from bandpy import runners, env, controllers, agents, utils, _arms


EPS = 0.1

# We first monkeypatch all the controller and propose a simple agent to
# to avoid unpredictible behaviour in term of estimation


def choose_agent(self):
    """Randomly return the name of an agent."""
    self.t += 1
    i = self.t % self.N
    self.T_i[i] += 1
    return f"agent_{i}"


def get_K(self):
    """Safely try to get the number of arms K."""
    msg = "The number of arms 'K' could not been retrieved."
    if hasattr(self, "K"):
        return self.K
    else:
        if hasattr(self, "arms"):
            if isinstance(self.arms, list):
                return len(self.arms)
            elif isinstance(self.arms, (_arms.LinearArms, _arms.QuadraticArms)):
                return len(self.arms._arms)
            else:
                raise RuntimeError(msg)
        else:
            raise RuntimeError(msg)


def pull_arm_ucb(self):
    """Return the best arm to pull following the UCB criterion."""
    K = self.get_K()
    return self.t % K


class SingleCluster_(controllers.SingleCluster):
    def choose_agent(self):
        return choose_agent(self)


class Decentralized_(controllers.Decentralized):
    def choose_agent(self):
        return choose_agent(self)


class OracleClustering_(controllers.OracleClustering):
    def choose_agent(self):
        return choose_agent(self)


class DynUCB_(controllers.DynUCB):
    def choose_agent(self):
        return choose_agent(self)

    def get_K(self):
        return get_K(self)

    def pull_arm_ucb(self, t, inv_A, theta):
        return pull_arm_ucb(self)


class CLUB_(controllers.CLUB):
    def choose_agent(self):
        return choose_agent(self)

    def get_K(self):
        return get_K(self)

    def pull_arm_ucb(self, t, inv_A, theta):
        return pull_arm_ucb(self)


class LBC_(controllers.LBC):
    def choose_agent(self):
        return choose_agent(self)

    def get_K(self):
        return get_K(self)

    def pull_arm_ucb(self, t, inv_A, theta):
        return pull_arm_ucb(self)


# We temporary crush the previous controller definition

controllers.SingleCluster = SingleCluster_
controllers.Decentralized = Decentralized_
controllers.OracleClustering = OracleClustering_
controllers.DynUCB = DynUCB_
controllers.CLUB = CLUB_
controllers.LBC = LBC_


class DebuggingLinearAgent(agents.LinUCB):
    def act(self, t):
        return t % self.K


@pytest.mark.parametrize("K", [5, 10])
@pytest.mark.parametrize(
    "d",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "T",
    [
        5000,
    ],
)
@pytest.mark.parametrize("N", [18, 32])
@pytest.mark.parametrize("lbda", [0.1, 1.0])
@pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_controllers_theta_estimation(K, d, T, N, lbda, alpha, seed):
    """Test the basic estimation functionality of all the controllers"""
    rng = utils.check_random_state(seed)
    sigma = 0.0
    n_thetas = 2
    A_init = lbda * np.eye(d)

    true_labels = []
    for label in range(n_thetas):
        true_labels += [label] * int(N / n_thetas)
    true_labels += [label] * (N - len(true_labels))

    true_agent_labels = dict()
    for i, l in enumerate(true_labels):
        true_agent_labels[f"agent_{i}"] = l

    thetas = [rng.randn(d, 1) for _ in range(n_thetas)]
    arms = [rng.randn(d, 1) for _ in range(K)]

    env_instance = env.ClusteredGaussianLinearBandit(
        d=d,
        N=N,
        T=T,
        arms=arms,
        thetas=thetas,
        theta_idx=true_labels,
        sigma=sigma,
        shuffle_labels=False,
        theta_offset=0.0,
        seed=seed,
    )

    all_controllers = {
        "0-Single": (
            controllers.SingleCluster,
            {
                "N": N,
                "agent_cls": DebuggingLinearAgent,
                "agent_kwargs": dict(arms=arms, alpha=alpha, lbda=lbda, seed=seed),
                "seed": seed,
            },
        ),
        "1-Ind": (
            controllers.Decentralized,
            {
                "N": N,
                "agent_cls": DebuggingLinearAgent,
                "agent_kwargs": dict(arms=arms, alpha=alpha, lbda=lbda, seed=seed),
                "seed": seed,
            },
        ),
        "2-Oracle": (
            controllers.OracleClustering,
            {
                "N": N,
                "agent_cls": DebuggingLinearAgent,
                "agent_kwargs": dict(arms=arms, alpha=alpha, lbda=lbda, seed=seed),
                "agent_labels": true_agent_labels,
                "seed": seed,
            },
        ),
        "3-DynUCB": (
            controllers.DynUCB,
            {
                "N": N,
                "n_clusters": n_thetas,
                "arms": env_instance.arms,
                "seed": seed,
                "alpha": alpha,
                "A_init": A_init,
            },
        ),
        "4-CLUB": (
            controllers.CLUB,
            {
                "N": N,
                "arms": env_instance.arms,
                "seed": seed,
                "alpha": alpha,
                "gamma": 1.0,
                "lbda": lbda,
                "A_init": A_init,
            },
        ),
        "5-LBC": (
            controllers.LBC,
            {
                "N": N,
                "arms": env_instance.arms,
                "seed": seed,
                "S": np.max([np.linalg.norm(theta) for theta in thetas]),
                "R": sigma,
                "lbda": lbda,
                "alpha": alpha,
                "A_init": A_init,
                "delta": 1e-3,
            },
        ),
    }

    for iterates in all_controllers.items():
        controller_name, (controller_cls, controller_kwargs) = iterates

        controller_instance = controller_cls(**controller_kwargs)

        results = runners.run_trials(
            env=env_instance,
            agent_or_controller=controller_instance,
            early_stopping=False,
            seeds=[seed],
            n_jobs=1,
            verbose=False,
        )

        # We propose simple tests on the estimation process

        for controller_instance, env_instance in results:
            # the controller's method .act() is called T - 1
            T_controller = controller_instance.t
            assert T_controller == (T - 1)

            sum_T_i = np.sum(controller_instance.T_i)
            assert sum_T_i == T

            if controller_name in ["2-Oracle", "3-DynUCB", "4-CLUB", "5-LBC"]:
                true_labels = [
                    env_instance.theta_per_agent[f"agent_{i}"] for i in range(N)
                ]
                labels = controller_instance.l_labels[-1]

                score_labels = metrics.rand_score(true_labels, labels)

                if controller_name == "2-Oracle":
                    assert score_labels == 1.0

            l_err_theta, l_err_theta_local = [], []
            for agent_name in controller_instance.agents.keys():
                true_theta = env_instance.thetas[
                    env_instance.theta_per_agent[agent_name]
                ]

                agent = controller_instance.agents[agent_name]
                theta_hat_local = agent.theta_hat_local
                theta_hat = agent.theta_hat

                err_theta_local = np.linalg.norm(theta_hat_local - true_theta)
                err_theta_local /= np.linalg.norm(true_theta)
                err_theta = np.linalg.norm(theta_hat - true_theta)
                err_theta /= np.linalg.norm(true_theta)

                l_err_theta_local.append(err_theta_local)
                l_err_theta.append(err_theta)

            mean_err_theta_local = np.mean(l_err_theta_local)
            mean_err_theta = np.mean(l_err_theta)

            assert mean_err_theta_local < EPS

            if controller_name == "0-Single":
                assert mean_err_theta >= mean_err_theta_local

            if controller_name == "1-Ind":
                assert mean_err_theta == mean_err_theta_local

            if controller_name == "2-Oracle":
                assert mean_err_theta <= mean_err_theta_local
