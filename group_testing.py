#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:45:50 2020

@author: ndexter
"""

from generate_groups import gen_measurement_matrix
from generate_individual_status import gen_status_vector
from generate_test_results import gen_test_vector
from model_preprocessing import problem_setup
from group_testing_optimizer import GT_optimizer
from group_testing_evaluation import decoder_evaluation
from group_testing_reporter import decoder_reporter
from model_preprocessing_with_pulp import *
import os
import decoder

# main method for testing
if __name__ == '__main__': 

    # options for setting up group testing problem
    opts = {}

    # unique run ID for prepending to file names
    opts['run_ID'] = 'debugging'

    # specify verbosity, plotting, and whether to generate MATLAB save files
    opts['verbose'] = False
    opts['plotting'] = False
    opts['saving'] = True

    # specify number of tests m and population size N
    opts['m'] = 300
    opts['N'] = 1000

    # specify infected individuals s
    opts['s'] = 300

    # specify the seed for initializing all of the random number generators
    opts['seed'] = 0

    # specify group size and maximum number of tests per individual
    opts['group_size'] = 16
    opts['max_tests_per_individual'] = 16

    # specify the graph generation method for generating the groups
    opts['graph_gen_method'] = 'no_multiple'

    # specify the noise model(s)
    #opts['test_noise_methods'] = ['threshold', 'binary_symmetric', 'permutation']
    opts['test_noise_methods'] = []

    for method in opts['test_noise_methods']:
        print('adding ' + method + ' noise', end=' ')
        if method == 'threshold':
            opts['theta_l'] = 0.00
            opts['theta_u'] = 0.10
            print('with theta_l = ' + str(opts['theta_l']) + ' and theta_u = ' + str(opts['theta_u']))
        elif method == 'binary_symmetric':
            opts['binary_symmetric_noise_prob'] = 0.26
            print('with binary_symmetric_noise_probability = ' + str(opts['binary_symmetric_noise_prob']))
        elif method == 'permutation':
            opts['permutation_noise_prob'] = 0.15
            print('with permutation_noise_probability = ' + str(opts['permutation_noise_prob']))

    # specify the file name for generating MATLAB save files
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'

    # specify parameters for decoding
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, r'problem.mps')

    # param = {}
    # param['file_path'] = file_path
    # param['lambda_w'] = 1
    # param['lambda_p'] = 100
    # param['lambda_n'] = 100
    # param['verbose'] = False
    # param['defective_num'] = None
    # param['sensitivity'] = None
    # param['specificity'] = None
    #
    # # specify CPLEX log
    # param['log_stream'] = None
    # param['error_stream'] = None
    # param['warning_stream'] = None
    # param['result_stream'] = None

    param = {}
    #param['file_path'] = file_path
    param['lambda_w'] = 1
    param['lambda_p'] = 100
    param['lambda_n'] = 100
    #param['verbose'] = False
    param['defective_num_lower_bound'] = None
    param['sensitivity_threshold'] = None
    param['specificity_threshold'] = None
    param['is_it_noiseless'] = True
    param['lp_relaxation'] = False
    # param['solver_name'] = 'CPLEX_PY'
    param['solver_options'] = {'timeLimit': 60, 'logPath': 'log.txt'}
    param['solver_name'] = 'COIN_CMD'

    # generate the measurement matrix from the given options
    A = gen_measurement_matrix(opts)

    # generate the infected status of the individuals
    u = gen_status_vector(opts)
    u = [i[0] for i in u]
    # generate the data corresponding to the group tests
    b = gen_test_vector(A, u, opts)
    #-------------------
    import numpy as np
    # generate the tests directly from A and u
    b_none = np.matmul(A, u)

    # rescale test results to 1
    b_none = np.minimum(b_none, 1)
    print('difference', len([i for i in range(len(b)) if b[i] != b_none[i]]))
    #-----------------------
    c = GroupTestingDecoder(**param)
    c.fit(A, b)
    print('SUM', np.sum(A, axis=0))
    ev_result = decoder_evaluation(u, c.solution())
