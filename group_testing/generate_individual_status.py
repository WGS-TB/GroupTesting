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

def gen_status_vector(seed=0, N=1000, s=10, verbose=False):
    """
    Function to generate the infected status of individuals (a vector)

    Parameters:
        seed (int): Seed for random number generation
        N (int): Population size
        s (int): Number of infected individuals
        verbose (bool): Flag for turning on debugging print statements

    Returns:
        u (binary numpy array): The status vector
    """

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
