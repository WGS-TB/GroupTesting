#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hzabeti
"""

from generate_groups import gen_measurement_matrix
from generate_individual_status import gen_status_vector
from generate_test_results import gen_test_vector
from model_preprocessing import problem_setup
from group_testing_optimizer import GT_optimizer
from group_testing_evaluation import decoder_evaluation
from group_testing_reporter import decoder_reporter
from multiprocessing import Pool
import numpy as np
import pandas as pd
import itertools
import os
import uuid


def multi_process_group_testing(opts, param):
    current_directory = os.getcwd()
    temp_unique_key = uuid.uuid1()
    for method in opts['test_noise_methods']:
        print('adding ' + method + ' noise', end=' ')
        if method == 'truncation':
            print('with no parameters, values in b = Au larger than 1 will be truncated to 1')
        elif method == 'threshold':
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
    file_path = os.path.join(current_directory, r'{}_problem.mps'.format(temp_unique_key))
    param['file_path'] = file_path

    # generate the measurement matrix from the given options
    A = gen_measurement_matrix(opts)

    # generate the infected status of the individuals
    u = gen_status_vector(opts)
    u = [i[0] for i in u]

    # generate the data corresponding to the group tests
    b = gen_test_vector(A, u, opts)

    # preparing ILP formulation
    problem_setup(A, b, param)
    print('Preparation is DONE!')

    # solve the system using decoder with CPLEX/Gurobi/GLPK
    sln = GT_optimizer(file_path=file_path, param=param, name="cplex")
    print('Decoding is DONE!')

    # remove the file
    os.remove(file_path)

    # evaluate the accuracy of the solution
    ev_result = decoder_evaluation(u, sln, opts['N'])
    print('Evaluation is DONE!')
    ev_result['m'] = opts['m']
    ev_result['N'] = opts['N']
    ev_result['s'] = opts['s']
    ev_result['seed'] = opts['seed']
    ev_result['group_size'] = opts['group_size']
    return ev_result

# main method for testing
if __name__ == '__main__':
    # options for setting up group testing problem
    opts =[{'run_ID': 'debugging', 'verbose': False, 'plotting': False, 'saving': True, 'm': 300, 'N': 600, 's': i,
            'seed': 0, 'group_size': 30, 'max_tests_per_individual': 15, 'graph_gen_method': 'no_multiple',
            'test_noise_methods': ['truncation']} for i in [30,40,50]]

    param = {'lambda_w': 1, 'lambda_p': 100, 'lambda_n': 100, 'verbose': False,
             'defective_num': None, 'sensitivity': None, 'specificity': None, 'log_stream': None, 'error_stream': None,
             'warning_stream': None, 'result_stream': None}

    with Pool(processes=16) as pool:
        results = pool.starmap(multi_process_group_testing, itertools.product(opts,[param]))
        pool.close()
        pool.join()
    column_names = ['N', 'm', 's', 'group_size', 'seed', 'tn', 'fp', 'fn', 'tp']
    pd.DataFrame(results).reindex(columns=column_names).to_csv('Results/CM.csv')

    # final report generation, cleanup, etc.

    # final output and end
