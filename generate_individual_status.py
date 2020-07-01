#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:45:50 2020

@author: ndexter
"""

import sys, math
import random
import numpy as np
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=60, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import scipy.io as sio

"""
Function to generate the infected status of individuals (a vector)
"""
def gen_status_vector(opts):

    random.seed(opts['seed'])

    # generate a random vector having sparsity level s
    # FIXME
    #indices = np.random.randint(opts['N'],size=(opts['s'],1))
    indices = random.sample(range(opts['N']), opts['s'])
    u = np.zeros((opts['N'],1))
    for i in indices:
        u[i] = 1

    try:
        assert np.sum(u) == opts['s']
    except AssertionError:
        errstr = ("Assertion Failed: opts['s'] = " + str(opts['s']) \
            + ", since sum(u) = " + str(np.sum(u))) 
        print(errstr)
        sys.exit()

    # save data to a MATLAB ".mat" file
    if opts['saving']:
        data = sio.loadmat(opts['data_filename'])
        data['u'] = u
        data['opts'] = opts
        data['seed'] = opts['seed']
        sio.savemat(opts['data_filename'], data)

    # return the vector, where the nth component represents the infected 
    # status of the nth individual
    return u

if __name__ == '__main__':

    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['N'] = 500
    opts['s'] = 20
    opts['verbose'] = True #False
    opts['plotting'] = True #False
    opts['saving'] = True
    opts['run_ID'] = 'GT_status_vector_generation_component'
    opts['seed'] = 0

    u = gen_status_vector(opts)

    # print shape of matrix
    if opts['verbose']:
        print("Generated status vector of size:")
        print(u.shape)
