#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hzabeti
"""

from generate_groups import gen_measurement_matrix
from generate_individual_status import gen_status_vector
from group_testing_evaluation import decoder_evaluation
from model_preprocessing_with_pulp import *
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import itertools
import os
import datetime
import time
import yaml
from shutil import copyfile
from utils import config_decoder, path_generator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.metrics import accuracy_score


def multi_process_group_testing(opts, param, lambda_selection):
    # for method in opts['test_noise_methods']:
    #     print('adding ' + method + ' noise', end=' ')
    #     if method == 'threshold':
    #         opts['theta_l'] = 0.00
    #         opts['theta_u'] = 0.10
    #         print('with theta_l = ' + str(opts['theta_l']) + ' and theta_u = ' + str(opts['theta_u']))
    #     elif method == 'binary_symmetric':
    #         opts['binary_symmetric_noise_prob'] = 0.26
    #         print('with binary_symmetric_noise_probability = ' + str(opts['binary_symmetric_noise_prob']))
    #     elif method == 'permutation':
    #         opts['permutation_noise_prob'] = 0.15
    #         print('with permutation_noise_probability = ' + str(opts['permutation_noise_prob']))
    # # specify the file name for generating MATLAB save files
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

        param['solver_options']['logPath'] = log_path + '/log_{}_{}_{}_{}_{}.txt'.format(opts['N'],
                                                                                         opts['group_size'],
                                                                                         opts['m'],
                                                                                         opts['s'],
                                                                                         opts['seed'])
        # TODO: quick fix change it later!
        if param['defective_num_lower_bound'] == 'p':
            param['defective_num_lower_bound'] = int(opts['s'])
        # --------------------------
        c = GroupTestingDecoder(**param)
        single_fit_start = time.time()
        if lambda_selection['cross_validation']:
            scoring = dict(Accuracy='accuracy',
                           balanced_accuracy=make_scorer(balanced_accuracy_score))
            grid = GridSearchCV(c, lambda_selection['param'], cv=lambda_selection['number_of_folds'],
                                refit='balanced_accuracy', scoring=scoring, n_jobs=-1,
                                return_train_score=True, verbose=10)
            grid.fit(A, b)
            c = grid.best_estimator_
            pd.DataFrame.from_dict(grid.cv_results_).to_csv(os.path.join(log_path,
                                                                         'cv_results_{}_{}_{}_{}_{}.csv'.format(
                                                                             opts['N'],
                                                                             opts['group_size'],
                                                                             opts['m'],
                                                                             opts['s'],
                                                                             opts['seed'])
                                                                         ))
            pd.DataFrame(grid.best_params_, index=[0]).to_csv(os.path.join(log_path,
                                                                           'best_params_{}_{}_{}_{}_{}.csv'.format(
                                                                               opts['N'],
                                                                               opts['group_size'],
                                                                               opts['m'],
                                                                               opts['s'],
                                                                               opts['seed'])
                                                                           ))
        else:
            c.fit(A, b)
        single_fit_end = time.time()
        print('SUM', np.sum(A, axis=0))
        print('Score:', c.score(A, b))
        # evaluate the accuracy of the solution
        ev_result = decoder_evaluation(u, c.solution())
        single_run_end = time.time()
        # TODO: this is only for cplex status! Change it to more general form!
        ev_result['Status'] = c.prob_.cplex_status
        ev_result['solver_time'] = round(single_fit_end - single_fit_start, 2)
        ev_result['time'] = round(single_run_end - single_run_start, 2)

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


# main method for testing
if __name__ == '__main__':
    start_time = time.time()

    # Read config file
    with open("config.yml", 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    # Load opts and param
    opts = config_decoder(config_dict['opts'])
    param = config_decoder(config_dict['param'])
    lambda_selection = config_dict['lambda_selection']
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
    copyfile('config.yml', os.path.join(result_path, 'config.yml'))

    with Pool(cpu_count()) as pool:
        results = pool.starmap(multi_process_group_testing, itertools.product(opts, param, lambda_selection))
        pool.close()
        pool.join()

    column_names = ['N', 'm', 's', 'group_size', 'seed', 'max_tests_per_individual', 'tn', 'fp', 'fn', 'tp',
                    'balanced_accuracy', 'Status', 'solver_time', 'time']
    # Saving files
    pd.DataFrame(opts).to_csv(os.path.join(result_path, 'opts.csv'))
    pd.DataFrame(results).reindex(columns=column_names).to_csv(os.path.join(result_path, 'CM.csv'))

    end_time = time.time()
    print(end_time - start_time)
    # final report generation, cleanup, etc.

    # final output and end
