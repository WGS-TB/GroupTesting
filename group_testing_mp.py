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
from utils import config_decoder,config_reader, path_generator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.metrics import accuracy_score


def multi_process_group_testing(design_param, decoder_param):
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
    design_param['data_filename'] = design_param['run_ID'] + '_generate_groups_output.mat'

    try:
        single_run_start = time.time()
        # generate the measurement matrix from the given options
        if design_param['generate_groups']:
            A = gen_measurement_matrix(design_param)
        else:
            pass
            # TODO: read from file
        # generate the infected status of the individuals
        if design_param['generate_individual_status']:
            u = gen_status_vector(design_param)
            u = [i[0] for i in u]
        else:
            pass
            # TODO: read from file
        # generate the data corresponding to the group tests
        if design_param['generate_test_results']:
            b = gen_test_vector(A, u, design_param)
        else:
            pass
            # TODO: read from file
        decoder_param['solver_options']['logPath'] = log_path + '/log_{}_{}_{}_{}_{}.txt'.format(design_param['N'],
                                                                                                 design_param['group_size'],
                                                                                                 design_param['m'],
                                                                                                 design_param['s'],
                                                                                                 design_param['seed'])
    except Exception as e:
        print(e)
    try:
        # TODO: quick fix change it later!
        # if decoder_param['defective_num_lower_bound'] == 'p':
        #     decoder_param['defective_num_lower_bound'] = int(design_param['s'])
        # --------------------------
        if decoder_param['decode']:
            c = GroupTestingDecoder(**decoder_param)
            single_fit_start = time.time()
            if decoder_param['lambda_selection']:
                scoring = dict(Accuracy='accuracy',
                               balanced_accuracy=make_scorer(balanced_accuracy_score))
                print('cross validation')
                grid = GridSearchCV(estimator=c, param_grid=decoder_param['cv_param'], cv=decoder_param['number_of_folds'],
                                    refit=decoder_param['eval_metric'], scoring=scoring, n_jobs=-1,
                                    return_train_score=True, verbose=10)
                grid.fit(A, b)

                print('fit')
                c = grid.best_estimator_
                pd.DataFrame.from_dict(grid.cv_results_).to_csv(os.path.join(log_path,
                                                                             'cv_results_{}_{}_{}_{}_{}.csv'.format(
                                                                                 design_param['N'],
                                                                                 design_param['group_size'],
                                                                                 design_param['m'],
                                                                                 design_param['s'],
                                                                                 design_param['seed'])
                                                                             ))
                pd.DataFrame(grid.best_params_, index=[0]).to_csv(os.path.join(log_path,
                                                                               'best_params_{}_{}_{}_{}_{}.csv'.format(
                                                                                   design_param['N'],
                                                                                   design_param['group_size'],
                                                                                   design_param['m'],
                                                                                   design_param['s'],
                                                                                   design_param['seed'])
                                                                               ))
            else:
                c.fit(A, b)
            single_fit_end = time.time()
            print('SUM', np.sum(A, axis=0))
            print('Score:', c.score(A, b))
            # TODO: save solution
            # evaluate the accuracy of the solution
        if decoder_param['evaluation']:
            try:
                if decoder_param['decode']:
                    ev_result = decoder_evaluation(u, c.solution())
                    ev_result['solver_time'] = round(single_fit_end - single_fit_start, 2)
                    # TODO: this is only for cplex status! Change it to more general form!
                    ev_result['Status'] = c.prob_.cplex_status
                else:
                    # TODO: read from file
                    pass
                # TODO: what if we only use decoder the we wouldn't have design_param
                ev_result['m'] = design_param['m']
                ev_result['N'] = design_param['N']
                ev_result['s'] = design_param['s']
                ev_result['seed'] = design_param['seed']
                ev_result['group_size'] = design_param['group_size']
                ev_result['max_tests_per_individual'] = design_param['max_tests_per_individual']
                print('Evaluation is DONE!')
            except Exception as e:
                print(e)
                ev_result = {'tn': None, 'fp': None, 'fn': None, 'tp': None}
            single_run_end = time.time()
            ev_result['time'] = round(single_run_end - single_run_start, 2)
        return ev_result
        # evaluate the accuracy of the solution
        # ev_result = decoder_evaluation(u, sln, opts['N'])
    except Exception as e:
        print(e)

    # ev_result['delta'] = opts['delta']
    # ev_result['rho'] = opts['rho']


# main method for testing
if __name__ == '__main__':
    start_time = time.time()

    # # Read config file
    # with open("config.yml", 'r') as config_file:
    #     config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    # # Load opts and param
    # opts = config_decoder(config_dict['opts'])
    # param = config_decoder(config_dict['param'])
    # lambda_selection = param['lambda_selection']
    design_param, decoder_param = config_reader('config.yml')
    print(design_param,decoder_param)
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
    if not decoder_param[0]['lambda_selection']:
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(multi_process_group_testing, itertools.product(design_param, decoder_param))
            pool.close()
            pool.join()
    else:
        results = [multi_process_group_testing(i[0],i[1]) for i in itertools.product(design_param, decoder_param)]

    column_names = ['N', 'm', 's', 'group_size', 'seed', 'max_tests_per_individual', 'tn', 'fp', 'fn', 'tp',
                    'balanced_accuracy', 'Status', 'solver_time', 'time']
    # Saving files
    pd.DataFrame(design_param).to_csv(os.path.join(result_path, 'opts.csv'))
    print("---------------------->",results)
    pd.DataFrame(results).reindex(columns=column_names).to_csv(os.path.join(result_path, 'CM.csv'))

    end_time = time.time()
    print(end_time - start_time)
    # final report generation, cleanup, etc.

    # final output and end
