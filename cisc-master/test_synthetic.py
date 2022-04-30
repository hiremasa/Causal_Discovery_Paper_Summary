#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Here, we test the accuracy of various causal inference techniques in identifying ANMs with different ranges of noise term.
"""
from __future__ import division
from functools import partial
import random
import sys
import time

import numpy as np
from stat_tests import friedman_test, nemenyi_multitest, bonferroni_dunn_test

from cisc import cisc
from dc import dc
from dr import dr
from utils import progress, plot_multiline, reverse_argsort, dc_compat


__author__ = "Kailash Budhathoki"
__email__ = "kbudhath@mpi-inf.mpg.de"
__copyright__ = "Copyright (c) 2017"
__license__ = "MIT"


class InfRes(object):

    def __init__(self):
        self.ncorrect = 0
        self.nwrong = 0
        self.nundec = 0

    def to_str(self, nsample):
        assert self.ncorrect + self.nwrong + self.nundec == nsample
        return str.center("%d %d %d %.2f" % (self.ncorrect, self.nwrong, self.nundec, self.ncorrect / nsample), 20)


def map_randomly(Xd, fXd):
    f = dict()
    for x in Xd:
        y = random.choice(fXd)
        f[x] = y
    return f


def generate_X(src, size):
    if src == "uniform":
        maxX = random.randint(1, 10)
        X = [random.randint(1, maxX) for i in xrange(size)]
    elif src == "multinomial":
        p_nums = [random.randint(1, 10) for i in xrange(random.randint(1, 11))]
        p_vals = [v / sum(p_nums) for v in p_nums]
        X = np.random.multinomial(size, p_vals, size=1)[0].tolist()
        X = [[i + 1] * f for i, f in enumerate(X)]
        X = [j for sublist in X for j in sublist]
    elif src == "binomial":
        n = random.randint(1, 40)
        p = random.uniform(0.1, 0.9)
        X = np.random.binomial(n, p, size).tolist()
    elif src == "geometric":
        p = random.uniform(0.1, 0.9)
        X = np.random.geometric(p, size).tolist()
    elif src == "hypergeometric":
        ngood = random.randint(1, 41)
        nbad = random.randint(1, 41)
        nsample = random.randint(1, min(41, ngood + nbad))
        X = np.random.hypergeometric(ngood, nbad, nsample, size).tolist()
    elif src == "poisson":
        lam = random.randint(1, 10)
        X = np.random.poisson(lam, size).tolist()
    elif src == "negativeBinomial":
        n = random.randint(1, 40)
        p = random.uniform(0.1, 0.9)
        X = np.random.negative_binomial(n, p, size).tolist()
    return X


def generate_additive_N(size):
    t = random.randint(1, 7)
    suppN = range(-t, t + 1)
    N = [random.choice(suppN) for i in xrange(size)]
    return N


def test_decision_rate():
    srcsX = ["uniform", "binomial", "negativeBinomial",
             "geometric", "hypergeometric", "poisson", "multinomial"]
    total = len(srcsX)
    count = 0
    progress(count, total)
    for srcX in srcsX:
        count += 1
        _decision_rate(srcX)
        progress(count, total)


def _decision_rate(srcX):
    nsample = 1000
    size = 1000
    level = 0.05
    suppfX = range(-7, 8)

    diff_cisc, diff_dc, diff_dr = [], [], []
    res_cisc, res_dc, res_dr = [], [], []

    for k in xrange(nsample):
        X = generate_X(srcX, size)
        suppX = list(set(X))
        f = map_randomly(suppX, suppfX)
        N = generate_additive_N(size)
        Y = [f[X[i]] + N[i] for i in xrange(size)]

        cisc_score = cisc(X, Y)
        dc_score = dc(dc_compat(X), dc_compat(Y))
        dr_score = dr(X, Y, level)

        undecided_cisc = cisc_score[0] == cisc_score[1]
        undecided_dc = dc_score[0] == dc_score[1]
        undecided_dr = (dr_score[0] < level and dr_score[1] < level) or (
            dr_score[0] > level and dr_score[1] > level)

        # todo(kailash): add the coin flip
        diff_cisc.append(abs(cisc_score[0] - cisc_score[1]))
        diff_dc.append(abs(dc_score[0] - dc_score[1]))
        diff_dr.append(abs(dr_score[0] - dr_score[1]))

        if undecided_cisc:
            res_cisc.append(random.choice([True, False]))
        else:
            res_cisc.append(cisc_score[0] < cisc_score[1])

        if undecided_dc:
            res_dc.append(random.choice([True, False]))
        else:
            res_dc.append(dc_score[0] < dc_score[1])

        if undecided_dr:
            res_dr.append(random.choice([True, False]))
        else:
            res_dr.append(dr_score[0] > level and dr_score[1] < level)

    indices_cisc = reverse_argsort(diff_cisc)
    indices_dc = reverse_argsort(diff_dc)
    indices_dr = reverse_argsort(diff_dr)

    diff_cisc = [diff_cisc[i] for i in indices_cisc]
    diff_dc = [diff_dc[i] for i in indices_dc]
    diff_dr = [diff_dr[i] for i in indices_dr]

    res_cisc = [res_cisc[i] for i in indices_cisc]
    res_dc = [res_dc[i] for i in indices_dc]
    res_dr = [res_dr[i] for i in indices_dr]

    dec_rate = np.arange(0.01, 1.01, 0.01)
    acc_cisc, acc_dc, acc_dr = [], [], []
    for r in dec_rate:
        dec_cisc_r = res_cisc[:int(r * nsample)]
        dec_dc_r = res_dc[:int(r * nsample)]
        dec_dr_r = res_dr[:int(r * nsample)]

        acc_cisc_r = sum(dec_cisc_r) / len(dec_cisc_r)
        acc_dc_r = sum(dec_dc_r) / len(dec_dc_r)
        acc_dr_r = sum(dec_dr_r) / len(dec_dr_r)

        acc_cisc.append(acc_cisc_r)
        acc_dc.append(acc_dc_r)
        acc_dr.append(acc_dr_r)
    # print dec_rate
    # print acc_dc
    # print acc_dr
    # print acc_cisc

    with open("results/dec_rate_synth_%s.dat" % srcX, "w") as writer:
        for i, r in enumerate(dec_rate):
            writer.write("%.2f %.2f %.2f %.2f\n" %
                         (r, acc_dc[i], acc_dr[i], acc_cisc[i]))
    # plot_multiline([acc_cisc, acc_dc, acc_dr], dec_rate, [
    #                "CISC", "DC", "DR"], "decision rate", "accuracy", "decision rate versus accuracy", "dec_rate_%sX.png" % srcX)


def test_accuracy():
    nsim = 5000
    size = 5000
    level = 0.05
    suppfX = range(-7, 8)
    srcsX = ["uniform", "binomial", "negativeBinomial",
             "geometric", "hypergeometric", "poisson", "multinomial"]
    print "-" * 48
    print "%18s%10s%10s%10s" % ("TYPE_X", "DC", "DR", "CISC")
    print "-" * 48

    fp = open("results/accuracy_synthetic.dat", "w")
    for srcX in srcsX:
        ncorrect_cisc, ncorrect_dc, ncorrect_dr = 0, 0, 0
        for k in xrange(nsim):
            X = generate_X(srcX, size)
            suppX = list(set(X))
            f = map_randomly(suppX, suppfX)
            N = generate_additive_N(size)
            Y = [f[X[i]] + N[i] for i in xrange(size)]

            cisc_score = cisc(X, Y)
            dc_score = dc(dc_compat(X), dc_compat(Y))
            dr_score = dr(X, Y, level)

            ncorrect_cisc += int(cisc_score[0] < cisc_score[1])
            ncorrect_dc += int(dc_score[0] < dc_score[1])
            ncorrect_dr += int(dr_score[0] > level and dr_score[1] < level)

        print "%18s%10.2f%10.2f%10.2f" % (srcX, ncorrect_dc / nsim, ncorrect_dr / nsim, ncorrect_cisc / nsim)
        fp.write("%s %.2f %.2f %.2f\n" % (srcX, ncorrect_dc /
                                          nsim, ncorrect_dr / nsim, ncorrect_cisc / nsim))
    print "-" * 48
    fp.close()


def test_accuracy_at_model_space():
    nsim = 500
    size = 1500
    level = 0.05
    suppfX = range(-7, 8)
    S = range(1, 7)
    srcsX = ["uniform", "binomial", "negativeBinomial",
             "geometric", "hypergeometric", "poisson", "multinomial"]

    print "%s | %s | %s | %s" % ("typeX".center(18), "sc_space".center(62), "dr_space".center(62), "full_space".center(62))
    print "%18s | %s %s %s | %s %s %s | %s %s %s" % ("", "sc".center(20), "dc".center(20), "dr".center(20), "sc".center(20), "dc".center(20), "dr".center(20), "sc".center(20), "dc".center(20), "dr".center(20))

    for srcX in srcsX:
        nsample_sc, nsample_anm, nsample_all = 0, 0, 0
        rsc_sc, rsc_dc, rsc_dr = InfRes(), InfRes(), InfRes()
        ranm_sc, ranm_dc, ranm_dr = InfRes(), InfRes(), InfRes()
        rall_sc, rall_dc, rall_dr = InfRes(), InfRes(), InfRes()

        for k in xrange(nsim):
            X = generate_X(srcX, size)
            suppX = list(set(X))
            f = map_randomly(suppX, suppfX)
            N = generate_additive_N(size)
            Y = [f[X[i]] + N[i] for i in xrange(size)]

            cisc_score = cisc(X, Y)
            dc_score = dc(dc_compat(X), dc_compat(Y))
            dr_score = dr(X, Y, level)

            undecided_dr = (dr_score[0] < level and dr_score[1] < level) or (
                dr_score[0] > level and dr_score[1] > level)

            if cisc_score[0] != cisc_score[1]:
                nsample_sc += 1
                rsc_sc.ncorrect += int(cisc_score[0] < cisc_score[1])
                rsc_sc.nwrong += int(cisc_score[0] > cisc_score[1])
                rsc_sc.nundec += int(cisc_score[0] == cisc_score[1])

                rsc_dc.ncorrect += int(dc_score[0] < dc_score[1])
                rsc_dc.nwrong += int(dc_score[0] > dc_score[1])
                rsc_dc.nundec += int(dc_score[0] == dc_score[1])

                rsc_dr.ncorrect += int(dr_score[0]
                                       > level and dr_score[1] < level)
                rsc_dr.nwrong += int(dr_score[0]
                                     < level and dr_score[1] > level)
                rsc_dr.nundec += int(undecided_dr)

            if not undecided_dr:
                nsample_anm += 1
                ranm_sc.ncorrect += int(cisc_score[0] < cisc_score[1])
                ranm_sc.nwrong += int(cisc_score[0] > cisc_score[1])
                ranm_sc.nundec += int(cisc_score[0] == cisc_score[1])

                ranm_dc.ncorrect += int(dc_score[0] < dc_score[1])
                ranm_dc.nwrong += int(dc_score[0] > dc_score[1])
                ranm_dc.nundec += int(dc_score[0] == dc_score[1])

                ranm_dr.ncorrect += int(dr_score[0]
                                        > level and dr_score[1] < level)
                ranm_dr.nwrong += int(dr_score[0]
                                      < level and dr_score[1] > level)
                ranm_dr.nundec += int(undecided_dr)

            nsample_all += 1
            rall_sc.ncorrect += int(cisc_score[0] < cisc_score[1])
            rall_sc.nwrong += int(cisc_score[0] > cisc_score[1])
            rall_sc.nundec += int(cisc_score[0] == cisc_score[1])

            rall_dc.ncorrect += int(dc_score[0] < dc_score[1])
            rall_dc.nwrong += int(dc_score[0] > dc_score[1])
            rall_dc.nundec += int(dc_score[0] == dc_score[1])

            rall_dr.ncorrect += int(dr_score[0]
                                    > level and dr_score[1] < level)
            rall_dr.nwrong += int(dr_score[0] < level and dr_score[1] > level)
            rall_dr.nundec += int(undecided_dr)

        print "%s | %s %s %s | %s %s %s | %s %s %s" % (srcX.center(18), rsc_sc.to_str(nsample_sc), rsc_dc.to_str(nsample_sc), rsc_dr.to_str(nsample_sc), ranm_sc.to_str(nsample_anm), ranm_dc.to_str(nsample_anm), ranm_dr.to_str(nsample_anm), rall_sc.to_str(nsample_all), rall_dc.to_str(nsample_all), rall_dr.to_str(nsample_all))


def test_accuracy_with_nonparam_tests():
    nsim = 250
    size = 1000
    level = 0.05
    fXd = range(-7, 8)
    S = range(1, 7)
    srcs = ["uniform", "binomial", "negativeBinomial",
            "geometric", "hypergeometric", "poisson", "multinomial"]

    print "%18s%10s%10s%10s" % ("X_TYPE", "ACC(CISC)", "ACC(DC)", "ACC(DR)")

    for src in srcs:
        ncorrect_this, nwrong_this, nind_this = 0, 0, 0
        ncorrect_dc, nwrong_dc, nind_dc = 0, 0, 0
        ncorrect_dr, nwrong_dr, nind_dr = 0, 0, 0
        diff_cisc, diff_dc, diff_dr = [], [], []

        for k in xrange(nsim):
            X = generate_X(src, size)
            Xd = list(set(X))
            f = map_randomly(Xd, fXd)
            t = random.choice(S)
            N = range(-t, t + 1)
            Y = [f[X[i]] + random.choice(N) for i in xrange(size)]

            cisc_score = cisc(X, Y)
            dc_score = dc(dc_compat(X), dc_compat(Y))
            dr_score = dr(X, Y, level)

            diff_cisc.append(abs(cisc_score[0] - cisc_score[1]))
            diff_dc.append(abs(dc_score[0] - dc_score[1]))
            diff_dr.append(abs(dr_score[0] - dr_score[1]))

            if cisc_score[0] < cisc_score[1]:
                ncorrect_this += 1
            elif cisc_score[0] > cisc_score[1]:
                nwrong_this += 1
            else:
                nind_this += 1

            if dc_score[0] < dc_score[1]:
                ncorrect_dc += 1
            elif dc_score[0] > dc_score[1]:
                nwrong_dc += 1
            else:
                nind_dc += 1

            if dr_score[0] > level and dr_score[1] < level:
                ncorrect_dr += 1

        # H0 = all the algorithms are equivalent
        iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(
            diff_cisc, diff_dc, diff_dr)
        if p_value < level:  # reject H0
            print "rejecting friedmans's H0"
            # H0 =  anking of CISC is different to each of the other methods
            ranks = dict([("cisc", rankings_cmp[0]),
                          ("dc", rankings_cmp[1]), ("dr", rankings_cmp[2])])
            comparisons, z_values, p_values, adj_p_values = bonferroni_dunn_test(
                ranks, "cisc")
            print comparisons, z_values, p_values, adj_p_values
        else:
            print "all the algorithms are the same"

        print "%18s%10.2f%10.2f%10.2f" % (src, ncorrect_this / nsim, ncorrect_dc / nsim, ncorrect_dr / nsim)


def execute_algorithms(X, Y, nloop):
    dcX, dcY = dc_compat(X), dc_compat(Y)
    dc_t, dr_t, cisc_t = 0, 0, 0
    for i in xrange(nloop):
        tstart = time.time()
        dc(dcX, dcY)
        tend = time.time()
        dc_t += tend - tstart

        tstart = time.time()
        dr(X, Y, 0.05)
        tend = time.time()
        dr_t += tend - tstart

        tstart = time.time()
        cisc(X, Y)
        tend = time.time()
        cisc_t += tend - tstart
    return dc_t / nloop, dr_t / nloop, cisc_t / nloop


def test_size_vs_runtime():
    nloop = 5
    suppX = range(20)
    suppY = range(20)
    sizes = range(1000000, 11000000, 1000000)
    fp = open("results/size_vs_runtime.dat", "w")
    for size in sizes:
        X = [random.choice(suppX) for i in xrange(size)]
        Y = [random.choice(suppY) for i in xrange(size)]
        dc_t, dr_t, cisc_t = execute_algorithms(X, Y, nloop)
        print size, dc_t, dr_t, cisc_t
        sys.stdout.flush()
        fp.write("%d %f %f %f\n" % (size, dc_t, dr_t, cisc_t))
    fp.close()


def test_domain_vs_runtime():
    nloop = 5
    size = 10000
    domains = range(100, 1100, 100)
    fp = open("results/domain_vs_runtime.dat", "w")
    for domain in domains:
        suppX = range(domain)
        suppY = range(domain)
        X = [random.choice(suppX) for i in xrange(size)]
        Y = [random.choice(suppY) for i in xrange(size)]
        dc_t, dr_t, cisc_t = execute_algorithms(X, Y, nloop)
        print domain, dc_t, dr_t, cisc_t
        sys.stdout.flush()
        fp.write("%d %f %f %f\n" % (domain, dc_t, dr_t, cisc_t))
    fp.close()


def test_power():
    sizes = [1000, 2000, 3000, 4000, 5000]
    level = 0.05
    srcX = "multinomial"
    nsim_cutoff, nsim_power = 100, 100
    suppfX = range(-7, 8)

    for size in sizes:
        diffs_dc, diffs_dr, diffs_cisc = [], [], []
        for i in xrange(nsim_cutoff):
            X = generate_X(srcX, size)
            suppX = list(set(X))
            f = map_randomly(suppX, suppfX)
            Y = [f[X[i]] for i in xrange(size)]

            dc_score = dc(dc_compat(X), dc_compat(Y))
            dr_score = dr(X, Y, level)
            cisc_score = cisc(X, Y)

            diffs_dc.append(abs(dc_score[0] - dc_score[1]))
            diffs_dr.append(abs(dr_score[0] - dr_score[1]))
            diffs_cisc.append(abs(cisc_score[0] - cisc_score[1]))

        cutoff_dc = np.percentile(diffs_dc, 5)
        cutoff_dr = np.percentile(diffs_dr, 5)
        cutoff_cisc = np.percentile(diffs_cisc, 5)
        print cutoff_dc, cutoff_dr, cutoff_cisc

        diffs_dc, diffs_dr, diffs_cisc = [], [], []
        for i in xrange(nsim_power):
            X = generate_X(srcX, size)
            suppX = list(set(X))
            f = map_randomly(suppX, suppfX)
            N = generate_additive_N(size)
            Y = [f[X[i]] + N[i] for i in xrange(size)]

            dc_score = dc(dc_compat(X), dc_compat(Y))
            dr_score = dr(X, Y, level)
            cisc_score = cisc(X, Y)

            diffs_dc.append(abs(dc_score[0] - dc_score[1]))
            diffs_dr.append(abs(dr_score[0] - dr_score[1]))
            diffs_cisc.append(abs(cisc_score[0] - cisc_score[1]))

        power_dc = sum(diffs_dc > cutoff_dc) * 1.0 / nsim_power
        power_dr = sum(diffs_dr > cutoff_dr) * 1.0 / nsim_power
        power_cisc = sum(diffs_cisc > cutoff_cisc) * 1.0 / nsim_power

        print size, power_dc, power_dr, power_cisc


if __name__ == "__main__":
    test_accuracy()
