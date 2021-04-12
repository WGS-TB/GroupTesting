#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hzabeti
"""

from generate_groups import gen_measurement_matrix
from generate_individual_status import gen_status_vector
from group_testing_evaluation import decoder_evaluation
from group_testing_decoder import *
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
from utils import config_decoder, config_reader, path_generator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.metrics import accuracy_score


def multi_process_group_testing(design_param, decoder_param):
    try:
        single_run_start = time.time()
        # generate the measurement matrix from the given options

        if design_param['generate_groups'] == 'alternative_module':
            generate_groups_alt_module = __import__(design_param['groups_alternative_module'][0], globals(), locals(),
                                                    [], 0)
            generate_groups_alt_function = getattr(generate_groups_alt_module,
                                                   design_param['groups_alternative_module'][1])
            A = generate_groups_alt_function(opts=design_param)
        elif design_param['generate_groups'] == 'input':
            A = np.genfromtxt(design_param['groups_input'], delimiter=',')
            design_param['m'], design_param['N'] = A.shape
            design_param['group_size'] = int(max(A.sum(axis=1)))
            design_param['max_tests_per_individual'] = int(max(A.sum(axis=0)))
        # elif design_param['generate_groups'] == 'generate':
        else:
            A = gen_measurement_matrix(opts=design_param)
        # generate the infected status of the individuals
        if design_param['generate_individual_status'] == 'input':
            u = np.genfromtxt(design_param['individual_status_input'], delimiter=',')
            design_param['s'] = np.count_nonzero(u)
        elif design_param['generate_individual_status'] == 'alternative_module':
            individual_status_alt_module = __import__(design_param['individual_status_alternative_module'][0],
                                                      globals(), locals(), [], 0)
            individual_status_alt_function = getattr(individual_status_alt_module,
                                                     design_param['individual_status_alternative_module'][1])
            u = individual_status_alt_function(opts=design_param)
        else:
            u = gen_status_vector(design_param)
            u = [i[0] for i in u]
        # generate the data corresponding to the group tests
        if design_param['generate_test_results'] == 'input':
            b = np.genfromtxt(design_param['test_results_input'], delimiter=',')
        elif design_param['generate_test_results'] == 'alternative_module':
            test_results_alt_module = __import__(design_param['test_results_alternative_module'][0],
                                                      globals(), locals(), [], 0)
            test_results_alt_function = getattr(test_results_alt_module,
                                                     design_param['test_results_alternative_module'][1])
            b = test_results_alt_function(opts=design_param)
        else:
            b = gen_test_vector(A, u, design_param)
        if design_param['save_to_file']:
            design_path = inner_path_generator(result_path, 'Design')
            design_matrix_path = inner_path_generator(design_path, 'Design_Matrix')
            pd.DataFrame(A).to_csv(report_file_path(design_matrix_path, 'design_matrix', design_param),
                                   header=None, index=None)
            individual_status_path = inner_path_generator(design_path,'Individual_Status')
            pd.DataFrame(u).to_csv(report_file_path(individual_status_path, 'individual_status', design_param),
                                   header=None, index=None)
            test_results_path = inner_path_generator(design_path,'Test_Results')
            pd.DataFrame(b).to_csv(report_file_path(test_results_path, 'test_results', design_param),
                                   header=None, index=None)

    except Exception as e:
        print(e)
    try:
        if decoder_param['decoder']:
            # TODO: this is only for cplex! Change it to more general form!
            decoder_param['solver_options']['logPath'] = report_file_path(log_path, 'log', design_param)
            c = GroupTestingDecoder(**decoder_param)
            single_fit_start = time.time()
            if decoder_param['lambda_selection']:
                scoring = dict(Accuracy='accuracy',
                               balanced_accuracy=make_scorer(balanced_accuracy_score))
                print('cross validation')
                grid = GridSearchCV(estimator=c, param_grid=decoder_param['cv_param'],
                                    cv=decoder_param['number_of_folds'],
                                    refit=decoder_param['eval_metric'], scoring=scoring, n_jobs=-1,
                                    return_train_score=True, verbose=10)
                grid.fit(A, b)

                print('fit')
                c = grid.best_estimator_
                pd.DataFrame.from_dict(grid.cv_results_).to_csv(report_file_path(log_path,'cv_results',design_param))
                pd.DataFrame(grid.best_params_, index=[0]).to_csv(report_file_path(log_path,'best_param',design_param))
            else:
                c.fit(A, b)
            single_fit_end = time.time()
            print('SUM', np.sum(A, axis=0))
            print('Score:', c.score(A, b))
            if design_param['save_to_file']:
                solution_path = inner_path_generator(result_path, 'Solutions')
                pd.DataFrame(c.solution()).to_csv(report_file_path(solution_path, 'solution', design_param),
                                                  header=None, index=None)
            # evaluate the accuracy of the solution
        if decoder_param['evaluation']:
            try:
                if decoder_param['decoder']:
                    ev_result = decoder_evaluation(u, c.solution())
                    ev_result['solver_time'] = round(single_fit_end - single_fit_start, 2)
                    # TODO: this is only for cplex status! Change it to more general form!
                    ev_result['Status'] = c.prob_.cplex_status
                else:
                    # TODO: read from file
                    pass
                # TODO: what if we only use decoder then we wouldn't have design_param
                print('Evaluation is DONE!')
            except Exception as e:
                print(e)
                ev_result = {'tn': None, 'fp': None, 'fn': None, 'tp': None}
            single_run_end = time.time()
            ev_result['time'] = round(single_run_end - single_run_start, 2)
            ev_result.update({key: design_param[key] for key in ['N', 'm', 's', 'group_size', 'seed',
                                                                 'max_tests_per_individual']})
        return ev_result
    except Exception as e:
        print(e)

    # ev_result['delta'] = opts['delta']
    # ev_result['rho'] = opts['rho']


# main method for testing
if __name__ == '__main__':
    start_time = time.time()
    # Read config file
    design_param, decoder_param = config_reader('config.yml')
    # output files path
    current_path, result_path = result_path_generator()
    log_path = inner_path_generator(result_path, 'Logs')
    if not decoder_param[0]['lambda_selection']:
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(multi_process_group_testing, itertools.product(design_param, decoder_param))
            pool.close()
            pool.join()
    else:
        results = [multi_process_group_testing(i[0], i[1]) for i in itertools.product(design_param, decoder_param)]

    column_names = ['N', 'm', 's', 'group_size', 'seed', 'max_tests_per_individual', 'tn', 'fp', 'fn', 'tp',
                    'balanced_accuracy', 'Status', 'solver_time', 'time']
    # Saving files
    pd.DataFrame(design_param).to_csv(os.path.join(result_path, 'opts.csv'))
    print("---------------------->", results)
    pd.DataFrame(results).reindex(columns=column_names).to_csv(os.path.join(result_path, 'CM.csv'))

    end_time = time.time()
    print(end_time - start_time)
    # final report generation, cleanup, etc.

    # final output and end
