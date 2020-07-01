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
Function to generate the results of the tests from the measurement matrix A 
and the status vector u
"""
def gen_test_vector(A, u, opts):

    random.seed(opts['seed'])

    # generate the tests directly from A and u
    b = np.matmul(A,u)

    # rescale test results to 1
    b = np.minimum(b,1)

    # save data to a MATLAB ".mat" file
    if opts['saving']:
        if path.exists(opts['data_filename']):
            data = sio.loadmat(opts['data_filename'])
        else:
            data = {}

        data['b'] = b
        sio.savemat(opts['data_filename'], data)

    # return the vector, where the nth component represents the infected 
    # status of the nth individual
    return b

if __name__ == '__main__':

    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['m'] = 300
    opts['N'] = 600
    opts['verbose'] = True #False
    opts['plotting'] = True #False
    opts['saving'] = True
    opts['run_ID'] = 'GT_test_result_vector_generation_component'
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'
    opts['seed'] = 0

    A = np.random.randint(2,size=(opts['m'],opts['N']))
    u = np.random.randint(2,size=opts['N'])
    b = gen_test_vector(A, u, opts)

    # print shape of matrix
    if opts['verbose']:
        print("Generated test result vector of size:")
        print(b.shape)
