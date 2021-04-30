#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hzabeti
"""

import argparse
from distutils.util import strtobool
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.metrics import accuracy_score
import sys
import pulp as pl

from group_testing import __version__
from group_testing.generate_groups import gen_measurement_matrix
from group_testing.generate_individual_status import gen_status_vector
from group_testing.generate_test_results import gen_test_vector
from group_testing.group_testing_evaluation import decoder_evaluation
from group_testing.group_testing_decoder import GroupTestingDecoder
import group_testing.utils as utils


def multi_process_group_testing(design_param, decoder_param):
    """
    Function to distribute parameters obtained from the config file to each module of the GroupTesting framework.

    Parameter:

        design_param (dict): Dictionary of parameters of design stage of the framework. i.e. generate_groups,
         generate_individual_status, generate_test_results.
        decoder_param (dict): Dictionary of parameters of decoding stage of the framework. i.e. group_testing_decoder,
         group_testing_evaluation.

    Returns:
        ev_result (dict): Dictionary of evaluation results.
    """
    try:
        single_run_start = time.time()
        # generate the measurement matrix from the given options
        if 'group_size' in design_param.keys() and str(design_param['group_size']).lower() == 'auto':
            assert 'N' in design_param.keys(), "To generate the group size automatically parameter 'N' is needed to be" \
                                               "defined in the config file"
            assert 's' in design_param.keys(), "To generate the group size automatically parameter 's' is needed to be" \
                                               "defined in the config file"
            assert design_param['s'] <= design_param['N'], " 's'> 'N': number of infected individuals can not be " \
                                                           "greater than number of individuals."
            design_param['group_size'] = utils.auto_group_size(design_param['N'], design_param['s'])
            #print("group size is {}".format(design_param['group_size']))
        if design_param['generate_groups'] == 'alternative_module':
            generate_groups_alt_module = __import__(design_param['groups_alternative_module'][0], globals(), locals(),
                                                    [], 0)
            generate_groups_alt_function = getattr(generate_groups_alt_module,
                                                   design_param['groups_alternative_module'][1])
            passing_param, remaining_param = utils.param_distributor(design_param, generate_groups_alt_function)
            A = generate_groups_alt_function(**passing_param)
        elif design_param['generate_groups'] == 'input':
            A = np.genfromtxt(design_param['groups_input'], delimiter=',')
            assert np.array_equal(A, A.astype(bool)), "The input design matrix A is not binary!"
            design_param['m'], design_param['N'] = A.shape
            design_param['group_size'] = int(max(A.sum(axis=1)))
            design_param['max_tests_per_individual'] = int(max(A.sum(axis=0)))
        elif design_param['generate_groups'] == 'generate':
            passing_param, remaining_param = utils.param_distributor(design_param, gen_measurement_matrix)
            A = gen_measurement_matrix(**passing_param)
        # generate the infected status of the individuals
        if design_param['generate_individual_status'] == 'input':
            u = np.genfromtxt(design_param['individual_status_input'], delimiter=',')
            assert u.size == design_param['N'], "Individual status input file does not have the correct size!"
            assert np.array_equal(u, u.astype(bool)), "Individual status input file is not binary!"
            design_param['s'] = np.count_nonzero(u)
        elif design_param['generate_individual_status'] == 'alternative_module':
            individual_status_alt_module = __import__(design_param['individual_status_alternative_module'][0],
                                                      globals(), locals(), [], 0)
            individual_status_alt_function = getattr(individual_status_alt_module,
                                                     design_param['individual_status_alternative_module'][1])
            passing_param, temp_remaining_param = utils.param_distributor(design_param, individual_status_alt_function)
            remaining_param.update(temp_remaining_param)
            u = individual_status_alt_function(**passing_param)
        elif design_param['generate_individual_status'] == 'generate':
            passing_param, temp_remaining_param = utils.param_distributor(design_param, gen_status_vector)
            remaining_param.update(temp_remaining_param)
            u = gen_status_vector(**passing_param)
            u = [i[0] for i in u]
        # generate the data corresponding to the group tests
        if design_param['generate_test_results'] == 'input':
            b = np.genfromtxt(design_param['test_results_input'], delimiter=',')
            assert b.size == design_param['m'], "Test results input file does not have the correct size!"
            assert np.array_equal(b, b.astype(bool)), "test results input file is not binary!"
        elif design_param['generate_test_results'] == 'alternative_module':
            test_results_alt_module = __import__(design_param['test_results_alternative_module'][0],
                                                 globals(), locals(), [], 0)
            test_results_alt_function = getattr(test_results_alt_module,
                                                design_param['test_results_alternative_module'][1])
            passing_param, temp_remaining_param = utils.param_distributor(design_param, test_results_alt_function)
            remaining_param.update(temp_remaining_param)
            b = test_results_alt_function(**passing_param)
        elif design_param['generate_test_results'] == 'generate':
            passing_param, temp_remaining_param = utils.param_distributor(design_param, gen_test_vector)
            remaining_param.update(temp_remaining_param)
            b = gen_test_vector(A, u, **passing_param)
        for main_param in ['N', 'm', 's', 'group_size', 'seed']:
            if main_param not in design_param:
                if main_param not in remaining_param:
                    design_param[main_param] = 'N\A'
                else:
                    design_param[main_param] = remaining_param[main_param]
        if 'save_to_file' in design_param.keys() and design_param['save_to_file']:
            design_path = utils.inner_path_generator(design_param['result_path'], 'Design')
            design_matrix_path = utils.inner_path_generator(design_path, 'Design_Matrix')
            pd.DataFrame(A).to_csv(utils.report_file_path(design_matrix_path, 'design_matrix', 'csv', design_param),
                                   header=None, index=None)
            if design_param['generate_individual_status']:
                individual_status_path = utils.inner_path_generator(design_path, 'Individual_Status')
                pd.DataFrame(u).to_csv(
                    utils.report_file_path(individual_status_path, 'individual_status', 'csv', design_param),
                    header=None, index=None)
            if design_param['generate_test_results']:
                test_results_path = utils.inner_path_generator(design_path, 'Test_Results')
                pd.DataFrame(b).to_csv(utils.report_file_path(test_results_path, 'test_results', 'csv', design_param),
                                       header=None, index=None)
    except:
        e = sys.exc_info()
        print("Error:", e)
        decoder_param['decoding'] = False

    if decoder_param['decoding']:
        try:
            if decoder_param['decoder'] == 'generate':
                if utils.dict_key_checker(decoder_param, 'solver_options') \
                        and utils.dict_key_checker(decoder_param['solver_options'], 'logPath') \
                        or decoder_param['lambda_selection']:
                    log_path = utils.inner_path_generator(design_param['result_path'], 'Logs')
                if utils.dict_key_checker(decoder_param, 'solver_options') \
                        and utils.dict_key_checker(decoder_param['solver_options'], 'logPath'):
                    if str(decoder_param['solver_options']['logPath']).lower() == 'auto':
                        decoder_param['solver_options']['logPath'] = utils.report_file_path(log_path, 'log', 'txt',
                                                                                            design_param)
                    else:
                        decoder_param['solver_options']['logPath'] = os.path.join(log_path,
                                                                                  decoder_param['solver_options'][
                                                                                      'logPath'])

                passing_param, _ = utils.param_distributor(decoder_param, GroupTestingDecoder)
                c = GroupTestingDecoder(**passing_param)
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

                    c = grid.best_estimator_
                    pd.DataFrame.from_dict(grid.cv_results_).to_csv(
                        utils.report_file_path(log_path,
                                               'cv_results', 'csv', design_param)
                    )
                    pd.DataFrame(grid.best_params_, index=[0]).to_csv(
                        utils.report_file_path(log_path,
                                               'best_param', 'csv', design_param)
                    )
                else:
                    c.fit(A, b)
                single_fit_end = time.time()
                # print('SUM', np.sum(A, axis=0))
                # print('Score:', c.score(A, b))
            elif decoder_param['decoder'] == 'alternative_module':
                # TODO: CV for alternative module. Is it needed?
                single_fit_start = time.time()
                decoder_alt_module = __import__(decoder_param['decoder_alternative_module'][0],
                                                globals(), locals(), [], 0)
                decoder_alt_function = getattr(decoder_alt_module,
                                               decoder_param['decoder_alternative_module'][1])
                passing_param, _ = utils.param_distributor(decoder_param, decoder_alt_function)
                c = decoder_alt_function(**passing_param)
                c.fit(A, b)
                single_fit_end = time.time()
            if 'save_to_file' in design_param.keys() and design_param['save_to_file']:
                solution_path = utils.inner_path_generator(design_param['result_path'], 'Solutions')
                pd.DataFrame(c.solution()).to_csv(
                    utils.report_file_path(solution_path, 'solution', 'csv', design_param),
                    header=None, index=None)
                # evaluate the accuracy of the solution
            if decoder_param['evaluation']:
                try:
                    ev_result = decoder_evaluation(u, c.solution(), decoder_param['eval_metric'])
                    ev_result['fitting_time'] = round(single_fit_end - single_fit_start, 2)
                    if 'cplex_status' in c.prob_.__dict__.keys():
                        ev_result['Status'] = c.prob_.cplex_status
                    else:
                        ev_result['Status'] = pl.LpStatus[c.prob_.status]
                except Exception as e:
                    print(e)
                    ev_result = {'tn': None, 'fp': None, 'fn': None, 'tp': None}
                single_run_end = time.time()
                ev_result['time'] = round(single_run_end - single_run_start, 2)
                ev_result.update({key: design_param[key] for key in ['N', 'm', 's', 'group_size', 'seed',
                                                                     'max_tests_per_individual', 'test_noise_methods',
                                                                     'permutation_noise_prob', 'theta_l',
                                                                     'theta_u', 'binary_symmetric_noise_prob']
                                  if key in design_param.keys()})
                return ev_result
        except Exception as e:
            print(e)
    else:
        print("Decoding was not performed!")


# main method for testing
def main(sysargs=sys.argv[1:]):
    """
    Main method for running the GroupTesting framework.

    Parameters:

        sysargs: args from command-line.
    """
    start_time = time.time()

    # argparse
    parser = argparse.ArgumentParser(prog='GroupTesting', description='Description')
    required_args = parser.add_argument_group('required arguments')
    parser.add_argument(
        '--version', action='version',
        version="%(prog)s version {version}".format(version=__version__)
    )
    required_args.add_argument(
        '--config', dest='config', metavar='FILE',
        help='Path to the config.yml file'
    )
    parser.add_argument(
        '--output-dir', dest='output_path', metavar='DIR',
        help='Path to the output directory',
    )
    parser.add_argument(
        '--parallel', dest='parallel', metavar='BOOL',
        help='whether to use multiprocessing. The default value is True.', default=True, type=bool
    )
    parser.add_argument(
        '--available-solvers', dest='available_solver', action='store_true',
        help='show available solvers for decoding and exit'
    )

    args = parser.parse_args()
    if args.available_solver:
        print("Available solvers for decoding are: {} ".format(pl.list_solvers(onlyAvailable=True)))
        parser.exit()
    if args.config is None:
        parser.error("the following arguments are required: --config")
    # Read config file
    design_param, decoder_param = utils.config_reader(args.config)
    # output files path
    current_path, result_path = utils.result_path_generator(args)
    for d in design_param:
        d['result_path'] = result_path
    if args.parallel \
            and not any([d['lambda_selection'] if 'lambda_selection' in d.keys() else False for d in decoder_param]):
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(multi_process_group_testing, itertools.product(design_param, decoder_param))
            pool.close()
            pool.join()
    else:
        results = [multi_process_group_testing(i[0], i[1]) for i in itertools.product(design_param, decoder_param)]

    # Saving files
    pd.DataFrame(design_param).to_csv(os.path.join(result_path, 'opts.csv'))
    if all(v is not None for v in results):
        print(pd.DataFrame(results))
        pd.DataFrame(results).to_csv(os.path.join(result_path, 'ConfusionMatrix.csv'))

    end_time = time.time()
    print(end_time - start_time)
