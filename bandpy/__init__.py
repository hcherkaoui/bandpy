""" Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from .info import __version__
from .runners import run_one_trial, run_trials, run_trials_with_grid_search


__version__ = __version__

__all__ = ['run_one_trial',
           'run_trials',
           'run_trials_with_grid_search',
           ]

MAX_K = 10000
