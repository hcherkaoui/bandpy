""" Simple example to demonstrate the overlapping ellipsoid test

Launch it with ::
    $ python

"""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.fr>

import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from bandpy import _compils, utils


plt.style.use('tableau-colorblind10')

t = 2.0

A = np.array([[1.0, 0.0], [0.0, 1.0]])
B = np.array([[1.0, 0.5], [0.5, 0.75]])

inv_A = np.linalg.inv(A)
cho_B = np.linalg.cholesky(B)

lambdas, phi = _compils.geigh(inv_A * t, cho_B / np.sqrt(t))

a = np.array([[0.0, 0.0]]).T
b0 = np.array([[1.0, 1.0]]).T
delta_b = np.array([[1.0, 1.0]]).T

bounds = [0.0 + 1e-6, 1.0 - 1e-6]

ncols, lim = 2, 2.0 * t

###############################################################################
# main
if __name__ == '__main__':

    t0_total = time.perf_counter()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-rows', type=int, default=5,
                        help='Number of rows of the plot.')
    parser.add_argument('--fig-fname', type=str,
                        default='0_ellipsoid_overlapping.pdf',
                        help='Figure filename.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbosity level.')
    args = parser.parse_args()


    fig, axis = plt.subplots(nrows=args.n_rows, ncols=ncols,
                             figsize=(4.0 * ncols, 4.0 * args.n_rows),
                             squeeze=False)

    l_alpha = np.linspace(0.0, 1.0, args.n_rows)
    for i, alpha in enumerate(l_alpha):

        b = b0 + alpha * delta_b

        v = phi.T.dot((a - b).ravel())

        X = np.array([a.ravel(), b.ravel()])
        offset = 5.0
        x_min, x_max = X[:, 0].min() - offset, X[:, 0].max() + offset
        y_min, y_max = X[:, 1].min() - offset, X[:, 1].max() + offset
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        # plot the ellipsoid
        f = lambda x : \
            (x.reshape((2, 1)) - a).T.dot(A).dot(x.reshape((2, 1)) - a)
        zz = np.array([f(x) for x in np.c_[xx.ravel(), yy.ravel()]])
        zz = zz.reshape(xx.shape)
        axis[i, 0].contour(xx, yy, zz, levels=[t])

        f = lambda x : \
            (x.reshape((2, 1)) - b).T.dot(B).dot(x.reshape((2, 1)) - b)
        zz = np.array([f(x) for x in np.c_[xx.ravel(), yy.ravel()]])
        zz = zz.reshape(xx.shape)
        axis[i, 0].contour(xx, yy, zz, levels=[t])

        axis[i, 0].set_xlim(-lim, lim)
        axis[i, 0].set_ylim(-lim, lim)
        value_K_min = _compils.K_min(inv_A * t, cho_B / np.sqrt(t), a, b)[1]
        axis[i, 0].set_title(f'Overlapping = {value_K_min > 0}')

        # plot the epigraph of the K func
        ss = np.linspace(bounds[0], bounds[1], 100)
        axis[i, 1].plot(ss, [_compils.K_f(s, lambdas, v) for s in ss], lw=2.0)
        axis[i, 1].axhline(0.0, c='k', lw=2.0)

        axis[i, 1].grid()
        axis[i, 1].set_ylabel('K(s)')
        axis[i, 1].set_xlabel('s')

    fig.tight_layout()

    if args.verbose:
        print(f"[Main] Saving plot under '{args.fig_fname}'")

    plt.savefig(args.fig_fname, dpi=300)

    ###########################################################################
    # Runtime
    delta_t = time.perf_counter() - t0_total
    if args.verbose:
        print(f"[Main] Script runs in {utils.format_duration(delta_t)}")