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
from model_preprocessing_with_pulp import *
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import itertools
import os
import uuid
import datetime
import time


def result_path_generator(noiseless, LP_relaxation, N, group_size, name):
    currentDate = datetime.datetime.now()
    dir_name = currentDate.strftime("%d") + '_' + currentDate.strftime("%b") + '_' + currentDate.strftime("%Y")
    if noiseless:
        noiseless = 'NoiseLess'
    else:
        noiseless = 'Noisy'
    if LP_relaxation:
        LP_relaxation = 'LP'
    else:
        LP_relaxation = 'ILP'
    local_path = "./Results/{}/{}/{}".format(dir_name, noiseless, LP_relaxation)
    path = os.getcwd()
    if not os.path.isdir(local_path):
        try:
            os.makedirs(path + local_path[1:])
        except OSError:
            print("Creation of the directory %s failed" % path + local_path[1:])
        else:
            print("Successfully created the directory %s " % path + local_path[1:])
    return path + local_path[1:] + "/{}_{}_{}_{}.csv".format(name, N, group_size, currentDate.strftime("%H:%M:%S"))


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
    #file_path = os.path.join(current_directory, r'{}_problem.mps'.format(temp_unique_key))
    #param['file_path'] = file_path

    try:
        # generate the measurement matrix from the given options
        A = gen_measurement_matrix(opts)
        # generate the infected status of the individuals
        u = gen_status_vector(opts)
        u = [i[0] for i in u]

        # generate the data corresponding to the group tests
        b = gen_test_vector(A, u, opts)

        c = GroupTestingDecoder(**param)
        c.fit(A, b)
        print('SUM', np.sum(A, axis=0))
        print('Score:', c.score(A, b))
        # evaluate the accuracy of the solution
        ev_result = decoder_evaluation(u, c.solution())

        # evaluate the accuracy of the solution
        # ev_result = decoder_evaluation(u, sln, opts['N'])
    except Exception as e:
        print(e)
        ev_result = {'tn': None, 'fp': None, 'fn': None, 'tp': None}
    print('Evaluation is DONE!')
    ev_result['m'] = opts['m']
    ev_result['N'] = opts['N']
    ev_result['s'] = opts['s']
    ev_result['seed'] = opts['seed']
    ev_result['group_size'] = opts['group_size']
    ev_result['max_tests_per_individual'] = opts['max_tests_per_individual']
    #ev_result['delta'] = opts['delta']
    #ev_result['rho'] = opts['rho']
    return ev_result


# main method for testing
if __name__ == '__main__':
    # options for setting up group testing problem
    start_time = time.time()
    seed_list = range(10)
    #rho_list = [0.1]
    N_list = [1000]
    prevalence_list = np.arange(0.01, 0.21, 0.01)
    group_size_list = [8]
    m_list = np.arange(0.01, 1.01, 0.01)
    divisibility_list = [8]

    # opts = [{'run_ID': 'debugging', 'verbose': False, 'plotting': False, 'saving': True, 'm': int((p + round(1 / g,
    # 3)) * N), 'N': N, 's': int((p + round(1 / g, 3)) * N * r), 'seed': seed, 'group_size': g,
    # 'max_tests_per_individual': d, 'graph_gen_method': 'no_multiple', 'test_noise_methods': ['truncation'],
    # 'delta': round(p + round(1 / g, 3), 3), 'rho': round(r, 3)} for seed in seed_list for N in N_list for g in
    # group_size_list for p in m_list for r in rho_list for d in divisibility_list]

    opts = [{'run_ID': 'debugging', 'verbose': False, 'plotting': False, 'saving': True,
             'm': int(m * N), 'N': N, 's': int(N * p),
             'seed': seed, 'group_size': g, 'max_tests_per_individual': d, 'graph_gen_method': 'no_multiple',
             'test_noise_methods': ['truncation']} for seed
            in seed_list for N in N_list for g in
            group_size_list
            for m in m_list for d in divisibility_list for p in prevalence_list]

    param = {'lambda_w': 1, 'lambda_p': 100, 'lambda_n': 100, 'fixed_defective_num': None,
             'sensitivity_threshold': None,
             'specificity_threshold': None, 'is_it_noiseless': True, 'lp_relaxation': False, 'solver_name': 'CPLEX_PY'}

    with Pool(cpu_count()) as pool:
        results = pool.starmap(multi_process_group_testing, itertools.product(opts, [param]))
        pool.close()
        pool.join()
    # column_names = ['N', 'm', 's', 'group_size', 'seed', 'delta', 'rho', 'tn', 'fp', 'fn', 'tp']
    column_names = ['N', 'm', 's', 'group_size', 'seed', 'max_tests_per_individual', 'tn', 'fp', 'fn', 'tp', 'balanced_accuracy']
    # Saving files
    opts_path = result_path_generator(param['is_it_noiseless'], param['lp_relaxation'], N_list[0], group_size_list[0],
                                      'opts')
    pd.DataFrame(opts).to_csv(opts_path)

    result_path = result_path_generator(param['is_it_noiseless'], param['lp_relaxation'], N_list[0], group_size_list[0],
                                        'CM')
    pd.DataFrame(results).reindex(columns=column_names).to_csv(result_path)
    end_time = time.time()
    print(end_time-start_time)
    # final report generation, cleanup, etc.

    # final output and end
