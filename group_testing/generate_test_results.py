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

from group_testing.generate_groups import gen_measurement_matrix
from group_testing.generate_individual_status import gen_status_vector
import group_testing.utils as utils

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=60, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))
import scipy.io as sio

def gen_test_vector(A, u, seed=0, m=100, verbose=False, test_noise_methods=[], binary_symmetric_noise_prob=0.26,
                    theta_l=0, theta_u=0.0625, permutation_noise_prob=0.01):
    """
    Function to generate the results of the tests from the measurement matrix A 
    and the status vector u in both the noiseless and noisy setting

    Parameters:
        A (binary numpy array): The group testing matrix
        u (binary numpy array): The individual status vector
        seed (int): Seed for random number generation
        m (int): Number of group tests
        verbose (bool): Flag for turning on debugging print statements
        test_noise_methods (str list): The list of noise models to apply
        binary_symmetric_noise_prob (float): The noise level for the binary symmetric noise model
        theta_l (float): The lower bound for the threshold noise model
        theta_u (float): The upper bound for the threshold noise model
        permutation_noise_prob (float): The noise level for the permutation noise model

    Returns:
        b (binary numpy array): The vector of results of the group tests
    """

    # set the seed used for test result noise
    random.seed(seed)
    np_random = np.random.RandomState(seed)

    # generate the tests directly from A and u
    b = np.matmul(A, u)

    if verbose:
        print('before minimum:')
        print(b)

    # rescale test results to 1
    b_temp = np.minimum(b, 1)

    if verbose:
        print('after minimum')
        print(np.c_[b, b_temp])

    # replace the original vector with the noisy vector
    b = np.array(b_temp)

    for method in test_noise_methods:

        if method == 'binary_symmetric':

            rho = binary_symmetric_noise_prob

            indices = np.arange(m)

            weight_vec = np.ones(m)
            weight_vec = weight_vec / np.sum(weight_vec)

            num_flip = math.ceil(m * rho)

            vec = np_random.choice(indices, size=num_flip, replace=False, p=weight_vec)

            # copy b into a new vector for adding noise
            b_noisy = np.array(b)

            if verbose:
                print('before flipping')
                print(b_noisy)

            # flip the resulting entries
            for v in vec:
                b_noisy[v] = (b_noisy[v] + 1) % 2

            if verbose:
                print('weight vector')
                print(weight_vec)
                print('indices to flip')
                print(vec)
                print('after flipping - left: b, right: b_noisy')
                print(np.c_[b, b_noisy])
                print('number of affected tests')
                print(np.sum(abs(b - b_noisy)))
                print('expected number')
                print(math.ceil(binary_symmetric_noise_prob * m))

            # replace the original vector with the noisy vector
            b = np.array(b_noisy)

        elif method == 'threshold':

            # find the group sizes
            Asum = np.sum(A, axis=1)

            print('sum is ' + str(Asum.shape))

            # copy b into a new vector for adding noise
            b_noisy = np.array(b)
            num_of_infected = np.matmul(A, u)
            if verbose:
                print('before threshold noise')
                print(b_noisy)

            # apply the threshold noise to b_noisy
            for i in range(m):

                if b_noisy[i] == 1:
                    if num_of_infected[i] / Asum[i] >= theta_u:
                        b_noisy[i] = 1
                    elif num_of_infected[i] / Asum[i] <= theta_l:
                        b_noisy[i] = 0
                    elif num_of_infected[i] / Asum[i] > theta_l and num_of_infected[i] / Asum[i] < theta_u:
                        # probability 1/2 of 0 or 1
                        # b_noisy[i] = np.random.randint(2)

                        # instead use probability of false negatives = 1/10 
                        b_noisy[i] = np_random.choice(np.arange(2), p=[0.1, 0.9])

            if verbose:
                print('after threshold noise - left: b, right: b_noisy')
                print(np.c_[b, b_noisy])

            # replace the original vector with the noisy vector
            b = np.array(b_noisy)

        elif method == 'permutation':
            # percentage of indices to permute
            rho = permutation_noise_prob

            # get the indices from which to select a subset to permute
            indices = np.arange(m)

            # permute all indices with equal probability 
            weight_vec = np.ones(m)
            weight_vec = weight_vec / np.sum(weight_vec)

            # determine the number of indices that need to be permuted
            num_permute = math.ceil(m * rho)

            # choose the indices to permute
            try:
                assert 2 * num_permute <= m
            except AssertionError as e:
                print('number of permuted items exceeds m, incorrect range for rho?')
                sys.exit()
            else:
                vec = np_random.choice(indices, size=2 * num_permute, replace=False, p=weight_vec)

            # copy b into a new vector for adding noise
            b_noisy = np.array(b)

            if verbose:
                print('before permuting')
                print(b_noisy)

            # find a permutation of the randomly selected indices
            # permuted_vec = np.random.permutation(vec)

            # print(vec)

            for i in range(num_permute):
                b_noisy[vec[i]] = b[vec[i + num_permute]]
                b_noisy[vec[i + num_permute]] = b[vec[i]]
                # print(np.c_[indices, b, b_noisy])

            # if verbose:
            # print('vector of indices to permute')
            # print(vec)
            # print('permutation of randomly selected indices')
            # print(permuted_vec)
            # print('original vector on indices to permute')
            # print(b_noisy[vec])
            # print('resulting permuted test results associated with those indices')
            # print(b_noisy[permuted_vec])

            # permute the original test results to add noise
            # b_noisy[vec] = b_noisy[permuted_vec]

            if verbose:
                print('after permuting - left: b, right: b_noisy')
                print(np.c_[b, b_noisy])

            # replace the original vector with the noisy vector
            b = np.array(b_noisy)
    return b


if __name__ == '__main__':
    """
    Main method for testing
    """

    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['m'] = 20
    opts['N'] = 400
    opts['s'] = math.ceil(0.04 * opts['N'])
    opts['run_ID'] = 'GT_test_result_vector_generation_component'
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'

    # noise types to test

    # opts['test_noise_methods'] = ['permutation']
    opts['test_noise_methods'] = []

    for method in opts['test_noise_methods']:
        print('adding ' + method + ' noise', end=' ')
        if method == 'threshold':
            opts['theta_l'] = 0.02
            opts['theta_u'] = 0.10
            print('with theta_l = ' + str(opts['theta_l']) + ' and theta_u = ' + str(opts['theta_u']))
        elif method == 'binary_symmetric':
            opts['binary_symmetric_noise_prob'] = 0.26
            print('with binary_symmetric_noise_probability = ' + str(opts['binary_symmetric_noise_prob']))
        elif method == 'permutation':
            opts['permutation_noise_prob'] = 0.15
            print('with permutation_noise_probability = ' + str(opts['permutation_noise_prob']))

    opts['seed'] = 0
    opts['group_size'] = 30
    opts['max_tests_per_individual'] = 15
    opts['graph_gen_method'] = 'no_multiple'
    opts['verbose'] = False
    opts['plotting'] = False
    opts['saving'] = False
    passing_param, _ = utils.param_distributor(opts, gen_measurement_matrix)
    A = gen_measurement_matrix(**passing_param)
    passing_param, _ = utils.param_distributor(opts, gen_status_vector)
    u = gen_status_vector(**passing_param)

    opts['verbose'] = True
    opts['plotting'] = False
    opts['saving'] = True
    b = gen_test_vector(A, u, opts)

    # print shape of matrix
    if opts['verbose']:
        print("Generated test result vector of size:")
        print(b.shape)
