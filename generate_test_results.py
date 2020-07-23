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
from generate_groups import gen_measurement_matrix
from generate_individual_status import gen_status_vector
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

    if opts['verbose']:
        print('A and u')
        print(A)
        print(u)
        print('before minimum:')
        print(b)

    if opts['test_noise_method'] == 'none':
        print('using noiseless testing model')

        # rescale test results to 1
        b = np.minimum(b,1)

        if opts['verbose']:
            print('after minimum')
            print(b)

    elif opts['test_noise_method'] == 'binary_symmetric':

        # rescale test results to 1
        b = np.minimum(b,1)

        if opts['verbose']:
            print('after minimum')
            print(b)

        rho = opts['test_noise_probability']

        indices = np.arange(opts['m'])

        weight_vec = np.ones(opts['m'])
        weight_vec = weight_vec/np.sum(weight_vec)

        num_flip = math.ceil(opts['m']*rho)

        vec = np.random.choice(indices, size=num_flip, replace=False, p=weight_vec)

        b_noisy = np.array(b)
        if opts['verbose']:
            print('before flipping - left: b, right: b_noisy')
            print(np.c_[b, b_noisy])

        # flip the resulting entries
        for v in vec:
            b_noisy[v] = (b_noisy[v] + 1) % 2

        # replace the original vector with the noisy vector
        b = np.array(b_noisy)

        if opts['verbose']:
            print('weight vector')
            print(weight_vec)
            print('indices to flip')
            print(vec)
            print('after flipping - left: b, right: b_noisy')
            print(np.c_[b, b_noisy])
            print('number of affected tests')
            print(np.sum(abs(b-b_noisy)))
            print('expected number')
            print(math.ceil(opts['test_noise_probability']*opts['m']))

    elif opts['test_noise_method'] == 'threshold':

        theta_l = opts['theta_l']
        theta_u = opts['theta_u']

        Asum = np.sum(A, axis = 1)
        for i in range(opts['m']):
            if b[i]/Asum[i] >= opts['theta_u']:
                b[i] = 1
            elif b[i]/Asum[i] <= opts['theta_l']:
                b[i] = 0
            elif b[i]/Asum[i] >= opts['theta_l'] and b[i]/Asum[i] <= opts['theta_u']:
                b[i] = np.random.randint(2)

        if opts['verbose']:
            print('after threshold noise')
            print(b)

    elif opts['test_noise_method'] == 'permutation':

        # rescale test results to 1
        b = np.minimum(b,1)

        if opts['verbose']:
            print('after minimum')
            print(b)

        # percentage of indices to permute
        rho = opts['test_noise_probability']

        # get the indices from which to select a subset to permute
        indices = np.arange(opts['m'])

        # permute all indices with equal probability 
        weight_vec = np.ones(opts['m'])
        weight_vec = weight_vec/np.sum(weight_vec)

        # determine the number of indices that need to be permuted
        num_permute = math.ceil(opts['m']*rho)

        # choose the indices to permute
        vec = np.random.choice(indices, size=num_permute, replace=False, p=weight_vec)

        # copy b into a new vector for adding noise
        b_noisy = np.array(b)

        if opts['verbose']:
            print('before permuting - left: b, right: b_noisy')
            print(np.c_[b, b_noisy])

        # find a permutation of the randomly selected indices
        permuted_vec = np.random.permutation(vec)

        if opts['verbose']:
            print(vec)
            print(permuted_vec)
            print(b[vec])
            print(b[permuted_vec])

        # permute the original test results to add noise
        b_noisy[vec] = b_noisy[permuted_vec]
        
        if opts['verbose']:
            print('after permuting - left: b, right: b_noisy')
            print(np.c_[b, b_noisy])

        # replace the original b with its noisy version
        b = np.array(b_noisy)

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
    opts['m'] = 20
    opts['N'] = 400
    opts['s'] = math.ceil(0.04*opts['N'])
    opts['run_ID'] = 'GT_test_result_vector_generation_component'
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'

    # parameters for binary symmetric noise
    #opts['test_noise_method'] = 'binary_symmetric'
    #opts['test_noise_probability'] = 0.26

    # parameters for threshold noise
    #opts['test_noise_method'] = 'threshold'
    #opts['theta_l'] = 0.05
    #opts['theta_u'] = 0.10

    # parameters for permutation noise
    opts['test_noise_method'] = 'permutation'
    opts['test_noise_probability'] = 0.15

    opts['seed'] = 0
    opts['group_size'] = 30
    opts['max_tests_per_individual'] = 15
    opts['graph_gen_method'] = 'no_multiple'
    opts['verbose'] = False
    opts['plotting'] = False
    opts['saving'] = False

    A = gen_measurement_matrix(opts)
    u = gen_status_vector(opts)

    opts['verbose'] = True
    opts['plotting'] = False
    opts['saving'] = True
    b = gen_test_vector(A, u, opts)

    # print shape of matrix
    if opts['verbose']:
        print("Generated test result vector of size:")
        print(b.shape)
