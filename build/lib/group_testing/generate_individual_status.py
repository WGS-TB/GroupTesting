#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:45:50 2020

@author: ndexter
"""

import sys, math
from os import path
import random
import numpy as np
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=60, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import scipy.io as sio

"""
Function to generate the infected status of individuals (a vector)
"""
def gen_status_vector(seed=0, N=1000, s=10, verbose=False):

    # set the seed used for status generation
    local_random = random.Random()
    local_random.seed(seed)
    np.random.seed(seed)

    # generate a random vector having sparsity level s
    indices = local_random.sample(range(N), s)
    u = np.zeros((N, 1))
    for i in indices:
        u[i] = 1

    try:
        assert np.sum(u) == s
    except AssertionError:
        errstr = ("Assertion Failed: opts['s'] = " + str(s) \
            + ", since sum(u) = " + str(np.sum(u))) 
        print(errstr)
        sys.exit()

    return u

if __name__ == '__main__':

    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['N'] = 500
    opts['s'] = 20
    opts['verbose'] = True #False
    # opts['plotting'] = True #False
    # opts['saving'] = True
    # opts['run_ID'] = 'GT_status_vector_generation_component'
    # opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'
    opts['seed'] = 0

    u = gen_status_vector(**opts)

    # print shape of matrix
    if opts['verbose']:
        print("Generated status vector of size:")
        print(u.shape)
