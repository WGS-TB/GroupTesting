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
import argparse

# main method for testing
if __name__ == '__main__': 

    # options for setting up group testing problem
    opts = {}
    parser = argparse.ArgumentParser()
    # unique run ID for prepending to file names
    parser.add_argument("--run_ID", default = 'debugging', type = str, help = "String for naming batch of trials in this run")
    parser.add_argument("--verbose", default = 0, type = int, help = "Switch for verbose output")
    parser.add_argument("--plotting", default = 0, type = int, help = "Switch for generating plots")
    parser.add_argument("--saving", default = 1, type = int, help = "Switch for generating MATLAB save files")
    parser.add_argument("--m", default = 300, type = int, help = "Number of group tests to use")
    parser.add_argument("--N", default = 1000, type = int, help = "Population size")
    parser.add_argument("--s", default = 300, type = int, help = "Number of infected individuals")
    parser.add_argument("--seed", default = 0, type = int, help = "Seed for random number generators")
    parser.add_argument("--group_size", default = 16, type = int, help = "Size of groups")
    parser.add_argument("--max_tests_per_individual", default = 16, type = int, help = "Maximum number of tests allowed per individual")
    parser.add_argument("--graph_gen_method", default = 'no_multiple', type = str, help = "Method for igraph to use in generating the groups")

    opts['run_ID'] = args.run_ID

    # specify verbosity, plotting, and whether to generate MATLAB save files
    opts['verbose'] = args.verbose
    opts['plotting'] = args.plotting
    opts['saving'] = args.saving

    # specify number of tests m and population size N
    opts['m'] = args.m
    opts['N'] = args.N

    # specify infected individuals s
    opts['s'] = args.s

    # specify the seed for initializing all of the random number generators
    opts['seed'] = args.seed

    # specify group size and maximum number of tests per individual
    opts['group_size'] = args.group_size
    opts['max_tests_per_individual'] = args.max_tests_per_individual

    # specify the graph generation method for generating the groups
    opts['graph_gen_method'] = args.graph_gen_method

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
    param['defective_num_lower_bound'] = 3
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
