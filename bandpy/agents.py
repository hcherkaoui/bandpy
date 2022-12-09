""" Agents module."""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from .linear_agents import LinUniform, LinUCB, EOptimalDesign, GreedyLinGapE
from .multi_armed_bandit_agents import FollowTheLeader, Uniform, EC, UCB


__all__ = ['LinUniform',
           'LinUCB',
           'EOptimalDesign',
           'GreedyLinGapE',
           'FollowTheLeader',
           'Uniform',
           'EC',
           'UCB',
           ]
