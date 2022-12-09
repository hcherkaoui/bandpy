""" Environment module."""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from .linear_bandit_env import (ClusteredGaussianLinearBandit, MovieLensEnv,
                                YahooEnv)
from .multi_armed_bandit_env import BernoulliKBandit, GaussianKBandit


__all__ = ['BernoulliKBandit',
           'GaussianKBandit',
           'ClusteredGaussianLinearBandit',
           'MovieLensEnv',
           'YahooEnv',
           ]
