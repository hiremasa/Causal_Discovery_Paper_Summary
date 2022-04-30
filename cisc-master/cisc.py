#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Causal inference on discrete data using stochastic complexity of multinomial.
"""
from collections import defaultdict
from sc import stochastic_complexity


__author__ = "Kailash Budhathoki"
__email__ = "kbudhath@mpi-inf.mpg.de"
__copyright__ = "Copyright (c) 2017"
__license__ = "MIT"


def marginals(X, Y):
    Ys = defaultdict(list)
    for i, x in enumerate(X):
        Ys[x].append(Y[i])
    return Ys


def cisc(X, Y):
    scX = stochastic_complexity(X)
    scY = stochastic_complexity(Y)

    mYgX = marginals(X, Y)
    mXgY = marginals(Y, X)

    scYgX = sum(stochastic_complexity(Z) for Z in mYgX.itervalues())
    scXgY = sum(stochastic_complexity(Z) for Z in mXgY.itervalues())

    ciscXtoY = scX + scYgX
    ciscYtoX = scY + scXgY

    return (ciscXtoY, ciscYtoX)


if __name__ == "__main__":
    import random
    from test_anm import map_randomly

    n = 1000
    Xd = range(1, 4)
    fXd = range(1, 4)
    f = map_randomly(Xd, fXd)
    N = range(-2, 3)

    X = [random.choice(Xd) for i in xrange(n)]
    Y = [f[X[i]] + random.choice(N) for i in xrange(n)]

    print cisc(X, Y)
