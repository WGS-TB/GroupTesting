#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:45:50 2020

@author: ndexter
"""

from generate_groups import gen_measurement_matrix
from generate_individual_status import gen_status_vector
from generate_test_results import gen_test_vector
import decoder

# main method for testing
if __name__ == '__main__': 

    # options for setting up group testing problem
    opts = {}

    # unique run ID for prepending to file names
    opts['run_ID'] = 'debugging'

    # specify number of tests m, population size N, sparsity level s
    opts['m'] = 300
    opts['N'] = 600
    opts['s'] = 30

    # specify the seed for initializing all of the random number generators
    opts['seed'] = 0

    # specify group size and maximum number of tests per individual
    opts['group_size'] = 30
    opts['max_tests_per_individual'] = 15

    # specify the graph generation method for generating the groups
    opts['graph_gen_method'] = 'no_multiple'

    # specify the noise model(s)
    #opts['test_noise_method'] = 'binary_symmetric'
    opts['test_noise_method'] = 'threshold'
    opts['test_noise_probability'] = 0.26
    opts['theta_l'] = 0.05
    opts['theta_u'] = 0.10

    # specify verbosity, plotting, and whether to generate MATLAB save files
    opts['verbose'] = False
    opts['plotting'] = False
    opts['saving'] = True

    # specify the file name for generating MATLAB save files
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'

    # generate the measurement matrix from the given options
    A = gen_measurement_matrix(opts)

    # generate the infected status of the individuals
    u = gen_status_vector(opts)

    # generate the data corresponding to the group tests
    b = gen_test_vector(A, u, opts)

    # solve the system using decoder with CPLEX

    # evaluate the accuracy of the solution

    # final report generation, cleanup, etc.

    # final output and end
