#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python wrapper for the matlab implementation of Causal Inference on Discrete Data using Additive Noise Models.
"""
import os
import StringIO
import sys

import matlab.engine


print "loading matlab engine ... ",
sys.stdout.flush()
cur_dir = os.path.dirname(__file__)
mengine = matlab.engine.start_matlab()
mengine.addpath(cur_dir)
out = StringIO.StringIO()
err = StringIO.StringIO()
print "\b[✓]"
sys.stdout.flush()


def dr(X, Y, level):
    # remove stdout arg for full debug in matlab
    fct_fw, p_fw, fct_bw, p_bw = mengine.fit_both_dir_discrete(
        X, 0, Y, 0, level, 0, nargout=4, stdout=out, stderr=err)
    return p_fw, p_bw


if __name__ == "__main__":
    import random
    X = [random.randint(1, 4) for i in xrange(1000)]
    Y = [X[i] + random.randint(1, 3) for i in xrange(1000)]
    print dr(X, Y, 0.05)
