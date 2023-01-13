""" Bandpy.

Short description:
------------------
Multi-armed single/multi-agent bandits Python package.

Description:
------------
Bandpy aims to provide all classical agents and controllers policies on a
various synthetic and real data environnments to ease benchmark for research
purposes.

Conventions:
------------
y : float, noisy reward observation.

no_noise_y : float, reward observation -without- noise.

s_t : array, [y_0, ..., y_t], noisy reward observation time serie.

S_t : array, [y_0, ..., sum_{s=1}^t y_s, ..., sum_{s=1}^T y_s], noisy
    cumulative reward observation time serie.

S_T : float, S_t[-1], sum_{s=1}^T y_s, noisy cumulative reward observation last
    value.

no_noise_s_t : array, [no_noise_y_0, ..., no_noise_y_t], reward -without- noise
    observation time serie.

no_noise_S_t : array, [no_noise_y_0, ..., sum_{s=1}^t no_noise_y_s, ...,
    sum_{s=1}^T no_noise_y_s], cumulative reward -without- noise observation
    time serie.

no_noise_S_T : float, no_noise_S_t[-1], sum_{s=1}^T no_noise_y_s, cumulative
    reward -without- noise observation last value.

best_s_t : array, [y_max, ..., y_max], best rewards time serie.

worst_s_t : array, [y_min, ..., y_min], worst rewards time serie.

best_S_t : array, [y_max * 1, ..., y_max * s, .., y_max * t], best cumulative
    rewards time serie.

worst_S_t : array, [y_min * 1, ..., y_min * s, .., y_min * t], worst cumulative
    rewards time serie.

best_S_T : float, y_max * T, best cumulative rewards last value.

worst_S_T : float, y_min * T, worst cumulative rewards last value.

r_t : array, [y_max - y_0, ..., y_max - y_t], noisy regret time serie.

R_t : array, [y_max - y_0, ..., sum_{s=1}^t y_max - y_s, ...,
    sum_{s=1}^T y_max - y_s], noisy cumulative regret time serie.

R_T : float, R_t[-1], sum_{s=1}^T y_max - y_s, noisy cumulative regret last
    value.

no_noise_no_noise_r_t : array, [y_max - no_noise_y_0, ...,
    y_max - no_noise_y_t], regret -without- noise time serie.

no_noise_R_t : array, [y_max - no_noise_y_0, ...,
    sum_{s=1}^t y_max - no_noise_y_s, ..., sum_{s=1}^T y_max - no_noise_y_s],
    cumulative regret -without- noise time serie.

no_noise_R_T : float, no_noise_R_t[-1], sum_{s=1}^T y_max - no_noise_y_s,
    cumulative regret -without- noise last value.
"""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

from .info import __version__


__version__ = __version__
