#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Here we test causal direction of univariate tübingen cause-effect pairs.
"""
from __future__ import division
import glob
import os
import re

import numpy as np

from cisc import cisc
from dc import dc
from dr import dr
from utils import dc_compat, plot_multiline, progress, reverse_argsort


XY_TRUTH_PATTERN = re.compile("x\s*-+\s*-*\s*>\s*y", re.IGNORECASE)
YX_TRUTH_PATTERN = re.compile("y\s*-+\s*-*\s*>\s*x", re.IGNORECASE)


def normalise(data):
    max_val = max(data)
    norm = lambda x: x * 1.0 / max_val
    return map(norm, data)


def get_ground_truths_of_tubingen_pairs():
    tubingen_pairs_dir = os.path.join(os.path.dirname(
        __file__), "data", "tubingen-pairs")
    tubingen_pairs = glob.glob(
        tubingen_pairs_dir + os.sep + "pair[0-9][0-9][0-9][0-9]_des.txt")

    truths = []
    for pair in tubingen_pairs:
        with open(pair, "r") as fp:
            desc = fp.read()
            if XY_TRUTH_PATTERN.findall(desc):
                truths.append(("X", "Y"))
            else:
                truths.append(("Y", "X"))
                # assert len(YX_TRUTH_PATTERN.findall(desc)) > 0
    return truths


def get_all_tubingen_pairs():
    tubingen_pairs_dir = os.path.join(os.path.dirname(
        __file__), "data", "tubingen-pairs")
    tubingen_pairs = glob.glob(
        tubingen_pairs_dir + os.sep + "pair[0-9][0-9][0-9][0-9].txt")

    return tubingen_pairs


def load_tubingen_pair(pair_path):
    data = np.loadtxt(pair_path)
    X = data[:, 0]
    Y = data[:, 1]
    return X, Y


def load_tubingen_pairs():
    tubingen_pairs = get_all_tubingen_pairs()
    for pair in tubingen_pairs:
        yield load_tubingen_pair(pair)


def preprocess_dc_style(X, scale_factor=20):
    if abs(max(X)) < 1:
        X = np.multiply(X, scale_factor)
    X = map(round, X)
    return X


def test_tubingen_pairs():
    epsilon = 0.0
    level = 0.05
    truths = get_ground_truths_of_tubingen_pairs()
    multivariate_pairs = [52, 53, 54, 55, 71]
    num_pairs = len(truths) - len(multivariate_pairs)

    num_correct = 0
    num_wrong = 0
    nsample = 0
    num_indecisive = 0
    res_cisc, res_dc, res_dr = [], [], []
    diffs_cisc, diffs_dc, diffs_dr = [], [], []

    progress(0, 95)
    for i, data in enumerate(load_tubingen_pairs()):
        if i + 1 in multivariate_pairs:
            continue

        X, Y = data
        # if i+1 in [65, 66, 67]:
        #     X, Y = preprocess_dc_style(X, 100), preprocess_dc_style(Y, 100)
        # else:
        #     X, Y = preprocess_dc_style(X), preprocess_dc_style(Y)
        # X, Y = normalise(X), normalise(Y)
        # discretizer = UnivariateIPDiscretizer(X, Y)
        # aX, X, aY, Y = discretizer.discretize()
        # cisc_score = cisc(Xd, Yd)
        nsample += 1
        cisc_score = cisc(X, Y)
        dc_score = cisc_score
        # dc_score = dc(dc_compat(X), dc_compat(Y))
        dr_score = dc_score
        # dr_score = dr(X.tolist(), Y.tolist(), level)
        print cisc_score
        diffs_cisc.append(abs(cisc_score[0] - cisc_score[1]))
        diffs_dc.append(abs(dc_score[0] - dc_score[1]))
        diffs_dr.append(abs(dr_score[0] - dr_score[1]))

        if cisc_score[0] < cisc_score[1]:
            cause_cisc = "X"
        elif cisc_score[0] > cisc_score[1]:
            cause_cisc = "Y"
        else:
            cause_cisc = ""

        if dc_score[0] < dc_score[1]:
            cause_dc = "X"
        elif dc_score[0] > dc_score[1]:
            cause_dc = "Y"
        else:
            cause_dc = ""

        if dr_score[0] > level and dr_score[1] < level:
            cause_dr = "X"
        elif dr_score[0] < level and dr_score[1] > level:
            cause_dr = "Y"
        else:
            cause_dr = ""

        true_cause = truths[i][0]
        if cause_cisc == "":
            res_cisc.append(random.choice([True, False]))
        elif cause_cisc == true_cause:
            res_cisc.append(True)
        else:
            res_cisc.append(False)

        if cause_dc == "":
            res_dc.append(random.choice([True, False]))
        elif cause_dc == true_cause:
            res_dc.append(True)
        else:
            res_dc.append(False)

        if cause_dr == "":
            res_dr.append(random.choice([True, False]))
        elif cause_dr == true_cause:
            res_dr.append(True)
        else:
            res_dr.append(False)

        progress(nsample, 95)

    # print "✓ = %3d    ✗ = %3d    ~ = %3d" % (num_correct, num_wrong,
    # num_indecisive)
    indices_cisc = reverse_argsort(diffs_cisc)
    indices_dc = reverse_argsort(diffs_dc)
    indices_dr = reverse_argsort(diffs_dr)

    diffs_cisc = [diffs_cisc[i] for i in indices_cisc]
    diffs_dc = [diffs_dc[i] for i in indices_dc]
    diffs_dr = [diffs_dr[i] for i in indices_dr]

    res_cisc = [res_cisc[i] for i in indices_cisc]
    res_dc = [res_dc[i] for i in indices_dc]
    res_dr = [res_dr[i] for i in indices_dr]

    dec_rate = np.arange(0.02, 1.01, 0.01)
    accs_cisc, accs_dc, accs_dr = [], [], []
    fp = open("results/dec_rate_benchmark.dat", "w")
    for r in dec_rate:
        maxIdx = int(r * nsample)
        rcisc = res_cisc[:maxIdx]
        rdc = res_dc[:maxIdx]
        rdr = res_dr[:maxIdx]

        accs_cisc.append(sum(rcisc) / len(rcisc))
        accs_dc.append(sum(rdc) / len(rdc))
        accs_dr.append(sum(rdr) / len(rdr))
        fp.write("%.2f %.2f %.2f %.2f\n" % (r, sum(rdc) / len(rdc),
                                          sum(rdr) / len(rdr), sum(rcisc) / len(rcisc)))
    fp.close()
    plot_multiline([accs_cisc, accs_dc, accs_dr], dec_rate, [
                   "CISC", "DC", "DR"], "decision rate", "accuracy", "decision rate versus accuracy")


if __name__ == "__main__":
    test_tubingen_pairs()
