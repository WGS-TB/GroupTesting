#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:45:50 2020

@author: ndexter
"""

# import standard libraries
import sys, math
from os import path
import random
import numpy as np

# np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=60, linewidth=100000,
                    formatter=dict(float=lambda x: "%.3g" % x))
import igraph
import scipy.io as sio

# import hdf5storage
# import matplotlib
# atplotlib.use('Agg')
# import unittest

# import plotting and data-manipulation tools
# import matplotlib.pyplot as plt




def gen_measurement_matrix(seed=0, N=1000, m=100, group_size=4, max_tests_per_individual=4, verbose=False,
                           graph_gen_method='no_multiple', plotting=False, saving=True, run_ID='debugging'):
    """
    Function to generate and return the group testing matrix based on near-doubly-regular design 

    Parameters:
        seed (int): Seed for random number generation
        N (int): Population size
        m (int): Number of group tests
        group_size (int): Size of the groups
        max_tests_per_individual (int): Maximum number of tests allowed per individual
        verbose (bool): Flag for turning on debugging print statements
        graph_gen_method (str): Method for igraph to use in generating the graph
        plotting (bool): Flag for turning on plotting 
        saving (bool): Flag for turning on saving the result
        run_ID (str): String for specifying name of run

    Returns:
        A (binary numpy array): The group testing matrix
    """

    # set the seed used for graph generation to the options seed
    random.seed(seed)
    np_random = np.random.RandomState(seed)

    # compute tests per individual needed to satisfy
    #           N*tests_per_individual == m*group_size
    avg_test_split = math.floor(m * group_size / N)
    if verbose:
        print('number of tests per individual needed to satisfy ' \
              + '(N*tests_per_individual == m*group_size) = ' + str(avg_test_split))

    # verify the given parameters can lead to a valid testing scheme:
    #   if floor(m*group_size/N) > max_tests_per_individual, we would need to 
    #   test each individual more times than the maximum allowed to satisfy 
    #   the group size constraint
    try:
        # ensure that we're not violating the maximum
        assert avg_test_split <= max_tests_per_individual
    except AssertionError:
        errstr = ('Assertion Failed: With group_size = ' + str(group_size) + ', since m = ' + str(m) +
                  ' and N = ' + str(N) + ' we have floor(m*group_size/N) = ' + str(avg_test_split) +
                  'which exceeds max_tests_per_individual = ' + str(max_tests_per_individual))
        print(errstr)
        sys.exit()

    # compute the actual tests per individual
    # note: we may end up with individuals being tested less than the maximum
    #   allowed number of tests, but this is not a problem (only the opposite is)
    tests_per_individual = max(avg_test_split, 1)

    if verbose:
        print("tests_per_individual = " + str(tests_per_individual))

    # In this model, the first N elements of the degree sequence correspond
    # to the population, while the last m elements correspond to the groups

    # in-degree corresponds to the individuals, here we set the first N 
    # entries of the in-degree sequence to specify the individuals 
    indeg = np.zeros(N + m)
    indeg[0:N] = tests_per_individual
    if N - m*group_size > 0:
        indeg[np.random.choice(np.arange(0,N), N - m*group_size, replace=False)] = 0

    if m*group_size - N*tests_per_individual > 0:
        temp = np.random.choice(np.arange(0,N), m*group_size - N*tests_per_individual, replace=False)
        indeg[temp] = indeg[temp] + 1

    # out-degree corresponds to the group tests, here we set the last m 
    # entries of the out-degree sequence to specify the groups
    outdeg = np.zeros(N + m)
    outdeg[N:(N + m)] = group_size

    # keep track of vertex types for final graph vertex coloring
    vtypes = np.zeros(N + m)
    vtypes[0:N] = 1

    # output the sum of indeg and outdeg if checking conditions
    if verbose:
        print("out degree sequence: {}".format(outdeg.tolist()))
        print("in degree sequence:  {}".format(indeg.tolist()))
        print("sum outdeg (groups) = {}".format(np.sum(outdeg)))
        print("sum indeg (individ) = {}".format(np.sum(indeg)))

    try:
        assert np.sum(outdeg) == np.sum(indeg)
    except AssertionError:
        errstr = ("Assertion Failed: Require sum(outdeg) = " + str(np.sum(outdeg)) + " == " \
                    + str(np.sum(indeg)) + " = sum(indeg)")
        print(errstr)
        print("out degree sequence: {}".format(outdeg.tolist()))
        print("in degree sequence: {}".format(indeg.tolist()))
        # sys.exit()

    # generate the graph
    try:
        assert igraph._igraph.is_graphical_degree_sequence(outdeg.tolist(), indeg.tolist())
        g = igraph.Graph.Degree_Sequence(outdeg.tolist(), indeg.tolist(), graph_gen_method)
        g.vs['vertex_type'] = vtypes
        assert np.sum(outdeg) == len(g.get_edgelist())
    except AssertionError:
        errstr = ("Assertion Failed: Require [sum(outdeg) = " + str(np.sum(outdeg)) + "] == [" \
                  + str(np.sum(indeg)) + " = sum(indeg)] == [ |E(G)| = " + str(len(g.get_edgelist())) + "]")
    except igraph._igraph.InternalError as err:
        print("igraph InternalError (likely invalid outdeg or indeg sequence): {0}".format(err))
        print("out degree sequence: {}".format(outdeg.tolist()))
        print("in degree sequence: {}".format(indeg.tolist()))
        sys.exit()
    except:
        print("Unexpected error:", sys.exec_info()[0])
        sys.exit()
    else:
        # get the adjacency matrix corresponding to the nodes of the graph
        A = np.array(g.get_adjacency()._get_data())
        if verbose:
            # print(g)
            # print(A)
            # print("before resizing")
            # print(A.shape)
            print("row sum {}".format(np.sum(A, axis=1)))
            print("column sum {}".format(np.sum(A, axis=0)))

        # the generated matrix has nonzeros in bottom left with zeros 
        # everywhere else, resize to it's m x N
        A = A[N:m + N, 0:N]
        # A = np.minimum(A, 1)

        # check if the graph corresponds to a bipartite graph
        check_bipartite = g.is_bipartite()

        # save the row and column sums
        row_sum = np.sum(A, axis=1)
        col_sum = np.sum(A, axis=0)

        # display properties of A and graph g
        if verbose:
            # print(A)
            # print("after resizing")
            # print(A.shape)
            print("row sum {}".format(row_sum))
            print("column sum {}".format(col_sum))
            print("max row sum {}".format(max(row_sum)))
            print("max column sum {}".format(max(col_sum)))
            print("min row sum {}".format(min(row_sum)))
            print("min column sum {}".format(min(col_sum)))
            print("g is bipartite: {}".format(check_bipartite))

        # set options and plot corresponding graph
        if plotting:
            layout = g.layout("auto")
            color_dict = {1: "blue", 0: "red"}
            g.vs['color'] = [color_dict[vertex_type] for vertex_type in g.vs['vertex_type']]
            B = g.vs.select(vertex_type='B')
            C = g.vs.select(vertex_type='C')
            visual_style = {}
            visual_style['vertex_size'] = 10
            visual_style['layout'] = layout
            visual_style['edge_width'] = 0.5
            visual_style['edge_arrow_width'] = 0.2
            visual_style['bbox'] = (1200, 1200)
            igraph.drawing.plot(g, **visual_style)

        # save data to a MATLAB ".mat" file
        data_filename = run_ID + '_generate_groups_output.mat'
        # if saving:
        #     data = {}
        #     # data['A'] = A
        #     data['bipartite'] = check_bipartite
        #     data['indeg'] = indeg
        #     data['outdeg'] = outdeg
        #     data['min_col_sum'] = min(col_sum)
        #     data['min_row_sum'] = min(row_sum)
        #     data['max_col_sum'] = max(col_sum)
        #     data['max_row_sum'] = max(row_sum)
        #     data['opts'] = opts
        #     sio.savemat(opts['data_filename'], data)

    # return the adjacency matrix of the graph
    return A