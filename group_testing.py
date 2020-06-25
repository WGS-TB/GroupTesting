#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:45:50 2020

@author: ndexter
"""

from generate_groups import gen_measurement_matrix
import decoder

# main method for testing
if __name__ == '__main__': 

    # specify number of tests m and population size N
    m = 300
    N = 600

    # specify group size and maximum number of tests per individual
    group_size = 30
    max_tests_per_individual = 15

    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['m'] = m
    opts['N'] = N
    opts['group_size'] = group_size
    opts['max_tests_per_individual'] = max_tests_per_individual
    opts['graph_gen_method'] = 'simple'
    opts['verbose'] = False
    opts['plotting'] = False
    opts['saving'] = True
    opts['run_ID'] = 'debugging'
    opts['seed'] = 0

    # generate the measurement matrix from the given options
    A = gen_measurement_matrix(opts)

    # generate the infected status of the individuals

    # generate the data corresponding to the group tests

    # solve the system using decoder with CPLEX

    # evaluate the accuracy of the solution

    # final report generation, cleanup, etc.

    # final output and end
