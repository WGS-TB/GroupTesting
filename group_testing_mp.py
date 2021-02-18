#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hzabeti
"""
from typing import Dict, List, Any, Union, Hashable

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
import yaml
from shutil import copyfile

# def result_path_generator(noiseless, LP_relaxation, N, group_size, name):
#     currentDate = datetime.datetime.now()
#     dir_name = currentDate.strftime("%d") + '_' + currentDate.strftime("%b") + '_' + currentDate.strftime("%Y")
#     if noiseless:
#         noiseless = 'NoiseLess'
#     else:
#         noiseless = 'Noisy'
#     if LP_relaxation:
#         LP_relaxation = 'LP'
#     else:
#         LP_relaxation = 'ILP'
#     local_path = "./Results/{}/{}/{}".format(dir_name, noiseless, LP_relaxation)
#     path = os.getcwd()
#     if not os.path.isdir(local_path):
#         try:
#             os.makedirs(path + local_path[1:])
#         except OSError:
#             print("Creation of the directory %s failed" % path + local_path[1:])
#         else:
#             print("Successfully created the directory %s " % path + local_path[1:])
#     return path + local_path[1:] + "/{}_{}_{}_{}.csv".format(name, N, group_size, currentDate.strftime("%H:%M:%S"))
def path_generator(file_path, file_name, file_format):
    currentDate = datetime.datetime.now()
    # TODO: change dir_name to unique code if you want to run multiple configs
    dir_name = currentDate.strftime("%b_%d_%Y_%H_%M")
    local_path = "./Results/{}".format(dir_name)
    path = os.getcwd()
    if not os.path.isdir(local_path):
        try:
            os.makedirs(path + local_path[1:])
        except OSError:
            print("Creation of the directory %s failed" % path + local_path[1:])
        else:
            print("Successfully created the directory %s " % path + local_path[1:])
    return path + local_path[1:] + "/{}.{}".format(file_name, file_format)

def multi_process_group_testing(opts, param):
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

    try:
        single_run_start = time.time()
        # generate the measurement matrix from the given options
        A = gen_measurement_matrix(opts)
        # generate the infected status of the individuals
        u = gen_status_vector(opts)
        u = [i[0] for i in u]

        # generate the data corresponding to the group tests
        b = gen_test_vector(A, u, opts)

        param['solver_options']['logPath'] = log_path+ '/log_{}_{}_{}_{}_{}.txt'.format(opts['N'],
                                                                                                   opts['group_size'],
                                                                                                   opts['m'], opts['s'],opts['seed'])
        #TODO: quick fix change it later!
        if param['defective_num_lower_bound'] == 'p':
            param['defective_num_lower_bound'] = int(opts['s'])
        #--------------------------
        c = GroupTestingDecoder(**param)
        single_fit_start = time.time()
        c.fit(A, b)
        single_fit_end = time.time()
        print('SUM', np.sum(A, axis=0))
        print('Score:', c.score(A, b))
        # evaluate the accuracy of the solution
        ev_result = decoder_evaluation(u, c.solution())
        single_run_end= time.time()
        # TODO: this is only for cplex status! Change it to more general form!
        ev_result['Status'] = c.prob_.cplex_status
        ev_result['solver_time'] = round(single_fit_end-single_fit_start,2)
        ev_result['time'] = round(single_run_end - single_run_start,2)

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
    # ev_result['delta'] = opts['delta']
    # ev_result['rho'] = opts['rho']
    return ev_result


def config_decoder(config_inpt):
    '''
    TODO: I used the following stackoverflow post for the return part:
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    '''
    for keys, vals in config_inpt.items():
        if isinstance(vals, dict):
            print(vals)
            if config_inpt[keys]['mode'] == 'continuous':
                config_inpt[keys] = np.arange(*config_inpt[keys]['values'])
            elif config_inpt[keys]['mode'] == 'discrete':
                config_inpt[keys] = config_inpt[keys]['values']
            elif config_inpt[keys]['mode'] == 'exact':
                config_inpt[keys].pop('mode')
                config_inpt[keys] = [config_inpt[keys]]
        else:
            config_inpt[keys] = [vals]
    return [dict(zip(config_inpt.keys(), vals)) for vals in itertools.product(*config_inpt.values())]


# main method for testing
if __name__ == '__main__':
    start_time = time.time()

    # Read config file
    with open("config.yml", 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    # Load opts and param
    opts = config_decoder(config_dict['opts'])
    param = config_decoder(config_dict['param'])
    # output files path
    path = os.getcwd()
    currentDate = datetime.datetime.now()
    dir_name = currentDate.strftime("%b_%d_%Y_%H_%M")
    result_path = os.path.join(os.getcwd(), "Results/{}".format(dir_name))
    log_path = os.path.join(result_path, "Logs")
    if not os.path.isdir(log_path):
        try:
            os.makedirs(log_path)
        except OSError:
            print("Creation of the directory %s failed" % log_path)
        else:
            print("Successfully created the directory %s " % log_path)
    # Copy config file

    copyfile('config.yml', os.path.join(result_path,'config.yml'))

    # # options for setting up group testing problem
    # seed_list = range(10)
    # # rho_list = [0.1]
    # N_list = [1000]
    # # prevalence_list = np.concatenate((np.arange(0.005,0.01,0.005),np.arange(0.01, 0.11, 0.01)))
    # prevalence_list = [0.05, 0.1]
    # group_size_list = [16]
    # m_list = np.arange(0.01, 1.01, 0.01)
    # divisibility_list = [16]
    #
    # # opts = [{'run_ID': 'debugging', 'verbose': False, 'plotting': False, 'saving': True, 'm': int((p + round(1 / g,
    # # 3)) * N), 'N': N, 's': int((p + round(1 / g, 3)) * N * r), 'seed': seed, 'group_size': g,
    # # 'max_tests_per_individual': d, 'graph_gen_method': 'no_multiple', 'test_noise_methods': [],
    # # 'delta': round(p + round(1 / g, 3), 3), 'rho': round(r, 3)} for seed in seed_list for N in N_list for g in
    # # group_size_list for p in m_list for r in rho_list for d in divisibility_list]
    #
    # opts = [{'run_ID': 'debugging', 'verbose': False, 'plotting': False, 'saving': True,
    #          'm': int(m * N), 'N': N, 's': int(N * p),
    #          'seed': seed, 'group_size': g, 'max_tests_per_individual': d, 'graph_gen_method': 'no_multiple',
    #          'test_noise_methods': []} for seed
    #         in seed_list for N in N_list for g in
    #         group_size_list
    #         for m in m_list for d in divisibility_list for p in prevalence_list]

    # param = {'lambda_w': 1, 'lambda_p': 100, 'lambda_n': 100, 'fixed_defective_num': None,
    #          'sensitivity_threshold': None,
    #          'specificity_threshold': None, 'is_it_noiseless': True, 'lp_relaxation': False, 'solver_name': 'CPLEX_PY',
    #          'solver_options': {'timeLimit': 600}}

    with Pool(cpu_count()) as pool:
        results = pool.starmap(multi_process_group_testing, itertools.product(opts, param))
        pool.close()
        pool.join()

    column_names = ['N', 'm', 's', 'group_size', 'seed', 'max_tests_per_individual', 'tn', 'fp', 'fn', 'tp',
                    'balanced_accuracy', 'Status','solver_time','time']
    # Saving files
    pd.DataFrame(opts).to_csv(os.path.join(result_path,'opts.csv'))
    pd.DataFrame(results).reindex(columns=column_names).to_csv(os.path.join(result_path,'CM.csv'))

    end_time = time.time()
    print(end_time - start_time)
    # final report generation, cleanup, etc.

    # final output and end
