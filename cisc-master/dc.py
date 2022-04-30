#!/usr/bin/env python
"""Implementation of Causal Inference on Discrete Data via Estimating
Distance Correlations.
Link: http://www.mitpressjournals.org/doi/pdf/10.1162/NECO_a_00820
"""
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy.spatial.distance import pdist, squareform

from dcor import dcor


__author__ = "Kailash Budhathoki"
__email__ = "kbudhath@mpi-inf.mpg.de"
__copyright__ = "Copyright (c) 2017"
__license__ = "MIT"


def dc(X, Y):
    prob_X, marg_X, prob_Y, marg_Y = distributions(X, Y)
    dXtoY = dcor(prob_X, marg_Y)
    dYtoX = dcor(prob_Y, marg_X)
    return (dXtoY, dYtoX)


def distributions(X, Y):
    N = len(X)
    unq_X = set(map(tuple, X))
    unq_Y = set(map(tuple, Y))
    idx = range(N)
    idx_X = dict(zip(unq_X, idx))
    idx_Y = dict(zip(unq_Y, idx))

    freq_XY = np.zeros((len(unq_X), len(unq_Y)))
    for i in range(N):
        ix = idx_X[tuple(X[i])]
        iy = idx_Y[tuple(Y[i])]
        freq_XY[ix, iy] += 1

    freq_X = np.sum(freq_XY, axis=1)[np.newaxis]
    freq_Y = np.sum(freq_XY, axis=0)[np.newaxis]
    prob_X = (freq_X / np.sum(freq_X)).transpose()
    prob_Y = (freq_Y / np.sum(freq_Y)).transpose()

    freqs_X = np.tile(freq_X.transpose(), (1, len(unq_Y)))
    freqs_Y = np.tile(freq_Y, (len(unq_X), 1))
    marg_X = (freq_XY / freqs_X).transpose()
    marg_Y = freq_XY / freqs_Y
    return prob_X, marg_X, prob_Y, marg_Y


if __name__ == "__main__":
    X = [[2, 3], [2, 3], [2, 4], [2], [2], [3], [3], [3, 4], [2, 3]]
    Y = [[1], [1], [1], [1], [0], [0], [0], [0], [1]]

    print dc(X, Y)
