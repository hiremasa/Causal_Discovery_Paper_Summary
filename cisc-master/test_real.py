#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests on real-world data.
"""
from __future__ import division
import csv
import os

import numpy as np

from cisc import cisc


__author__ = "Kailash Budhathoki"
__email__ = "kbudhath@mpi-inf.mpg.de"
__copyright__ = "Copyright (c) 2017"
__license__ = "MIT"


def test_nursery():
    print "testing cisc on nursery dataset"
    nursery_dir = os.path.join(os.path.dirname(__file__), "data", "nursery")
    nursery_dat_path = os.path.join(nursery_dir, "nursery.dat")

    X_labels = ["parents", "has_nurs",
                "family_form", "children", "housing", "finance", "social", "health"]
    Y_label = "application_evaluation"
    data = np.loadtxt(nursery_dat_path)
    nattr = data.shape[1]
    Y = data[:, nattr - 1]
    for i in xrange(nattr - 1):
        X = data[:, i]
        cisc_score = cisc(X, Y)
        if cisc_score[0] < cisc_score[1]:
            print "%s ⇒ %s" % (X_labels[i], Y_label),
        elif cisc_score[0] > cisc_score[1]:
            print "%s ⇐ %s" % (X_labels[i], Y_label),
        else:
            print "%s ~ %s" % (X_labels[i], Y_label),
        print
    print


def test_car():
    print "testing cisc on car dataset"
    car_dir = os.path.join(os.path.dirname(__file__), "data", "car")
    car_dat_path = os.path.join(car_dir, "car.dat")

    X_labels = ["buying price", "maintenance",
                "#dorrs", "capacity", "luggage boot", "safety"]
    Y_label = "car acceptibility"
    data = np.loadtxt(car_dat_path)
    nattr = data.shape[1]
    Y = data[:, nattr - 1]
    for i in xrange(nattr - 1):
        X = data[:, i]
        cisc_score = cisc(X, Y)
        if cisc_score[0] < cisc_score[1]:
            print "%s ⇒ %s" % (X_labels[i], Y_label),
        elif cisc_score[0] > cisc_score[1]:
            print "%s ⇐ %s" % (X_labels[i], Y_label),
        else:
            print "%s ~ %s" % (X_labels[i], Y_label),
        print
    print


def test_weather():
    print "testing cisc on weather dataset"
    weather_dir = os.path.join(os.path.dirname(__file__), "data", "weather")
    weather_dat_path = os.path.join(weather_dir, "weather.dat")

    Xs, Ys = [[]] * 4, [[]] * 4
    with open(weather_dat_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")

        for row in reader:
            assert len(row) == 8
            for i in xrange(4):
                Xs[i].append(row[i])

            for i in xrange(4):
                Ys[i].append(row[i + 4])

    assert len(Xs[0]) == len(Xs[1]) == len(Xs[2]) == len(Xs[3])
    assert len(Ys[0]) == len(Ys[1]) == len(Ys[2]) == len(Ys[3])

    for i in xrange(4):
        X = Xs[i]
        Y = Ys[i]
        cisc_score = cisc(X, Y)
        if cisc_score[0] < cisc_score[1]:
            print "X_%d ⇒ Y_%d" % (i, i)
        elif cisc_score[0] > cisc_score[1]:
            print "Y_%d ⇒ X_%d" % (i, i)
        else:
            print "X_%d ~ Y_%d" % (i, i)
    print


def test_adult():
    print "testing cisc on adult dataset"
    adult_dir = os.path.join(os.path.dirname(__file__), "data", "adult")
    adult_dat_path = os.path.join(adult_dir, "adult.dat")
    data = np.loadtxt(adult_dat_path)
    ncols = data.shape[1]
    income = map(int, data[:, ncols - 1])

    colnames = ["workclass", "education", "occupation"]
    for i in xrange(ncols - 1):
        X = map(int, data[:, i])
        cisc_score = cisc(X, income)
        if cisc_score[0] < cisc_score[1]:
            print "%s ⇒ %s" % (colnames[i], "income"),
        elif cisc_score[0] > cisc_score[1]:
            print "%s ⇐ %s" % (colnames[i], "income"),
        else:
            print "%s ~ %s" % (colnames[i], "income"),
        print cisc_score
    print


def test_abalone():
    # We do not discretise the variables just as what DR paper does
    print "testing cisc on abalone dataset"
    abalone_dir = os.path.join(os.path.dirname(__file__), "data", "abalone")
    abalone_dat_path = os.path.join(abalone_dir, "abalone.dat")
    data = np.loadtxt(abalone_dat_path)
    ncols = data.shape[1]
    sex = data[:, 0]

    colnames = ["Sex", "Length", "Diameter", "Height"]
    for i in xrange(1, ncols):
        Y = data[:, i]
        cisc_score = cisc(sex, Y)
        if cisc_score[0] < cisc_score[1]:
            print "%s ⇒ %s" % ("Sex", colnames[i]),
        elif cisc_score[0] > cisc_score[1]:
            print "%s ⇐ %s" % ("Sex", colnames[i]),
        else:
            print "%s ~ %s" % ("Sex", colnames[i]),
        print
    print


def test_acute():
    print "testing cisc on acute inflammation dataset"
    abalone_dir = os.path.join(os.path.dirname(__file__), "data", "acute")
    abalone_dat_path = os.path.join(abalone_dir, "acute.tsv")
    data = np.loadtxt(abalone_dat_path)

    diag1 = data[:, 6]
    diag2 = data[:, 7]

    colnames = ["temperature", "nausea", "lumber pain",
                "urine pushing", "micturition pains", "burning of urethra"]
    diagnoses = ["Inflammation of urinary bladder",
                 "Nephritis of renal pelvis origin "]
    for i in xrange(6):
        symptom = data[:, i]
        cisc_score = cisc(diag1, symptom)
        if cisc_score[0] < cisc_score[1]:
            print "%s ⇒ %s" % (diagnoses[0], colnames[i]),
        elif cisc_score[0] > cisc_score[1]:
            print "%s ⇐ %s" % (diagnoses[0], colnames[i]),
        else:
            print "%s ~ %s" % (diagnoses[0], colnames[i]),
        print cisc_score

        cisc_score = cisc(diag2, symptom)
        if cisc_score[0] < cisc_score[1]:
            print "%s ⇒ %s" % (diagnoses[1], colnames[i]),
        elif cisc_score[0] > cisc_score[1]:
            print "%s ⇐ %s" % (diagnoses[1], colnames[i]),
        else:
            print "%s ~ %s" % (diagnoses[1], colnames[i]),
        print cisc_score
    print


def test_faces():
    print "testing cisc on faces dataset"
    faces_dir = os.path.join(os.path.dirname(__file__), "data", "faces")
    faces_dat_path = os.path.join(faces_dir, "faces.tsv")
    data = np.loadtxt(faces_dat_path)
    parameter = data[:, 0]
    answer = data[:, 1]
    cisc_score = cisc(parameter, answer)
    if cisc_score[0] < cisc_score[1]:
        print "%s ⇒ %s" % ("parameter", "answer"),
    elif cisc_score[0] > cisc_score[1]:
        print "%s ⇐ %s" % ("parameter", "answer"),
    else:
        print "%s ~ %s" % ("parameter", "answer"),
    print cisc_score
    print


if __name__ == "__main__":
    test_faces()
    test_acute()
    test_abalone()
    test_car()
    test_nursery()
    # test_adult()
