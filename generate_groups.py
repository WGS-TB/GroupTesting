#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:45:50 2020

@author: ndexter
"""

# import standard libraries
import sys, math
import random
import numpy as np
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=60, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import igraph
import scipy.io as sio

#import hdf5storage
#import matplotlib
#atplotlib.use('Agg')
#import unittest

# import plotting and data-manipulation tools
#import matplotlib.pyplot as plt

#class TestMeasurementMatrix(unittest.TestCase):
#
#    def test_make_graph(self):
#        gen_measurement_matrix(opts)
        

# function to generate and return the matrix
def gen_measurement_matrix(opts):

    # set the seed used for graph generation to that passed in the options
    random.seed(opts['seed'])

    m = opts['m']
    N = opts['N']
    group_size = opts['group_size']
    max_tests_per_individual = opts['max_tests_per_individual']

    # tests must be less than half or will not be able to satisfy all constraints
    try:
        assert m <= math.ceil(N/2)
    except AssertionError:
        errstr = ("Assertion Failed: With group_size = " + str(group_size) \
            + " and max_tests_per_individual = " + str(max_tests_per_individual) \
            + ", m = " + str(m) + " must be less than N/2 = " + str(math.ceil(N/2)) \
            + " (or else some individuals will be tested more than the max times)")
        print(errstr)
        sys.exit()

    # tests we are able to run per individual
    try:
        assert math.floor(m*group_size/N) <= max_tests_per_individual
    except AssertionError:
        errstr = ("Assertion Failed: With group_size = " + str(group_size) \
            + ", since m = " + str(m) + " and N = " + str(N) \
            + " we have floor(m*group_size/N) = " + str(math.floor(m*group_size/N)) \
            + " which exceeds max_tests_per_individual = " + str(max_tests_per_individual)) 
        print(errstr)
        sys.exit()

    # compute the actual tests per individual
    # note: it may not be possible to satisfied the max_tests_per_individual 
    #       property based on the choice of m, N, and group size.
    tests_per_individual = min(max_tests_per_individual, math.floor(m*group_size/N))

    if opts['verbose']:
        print("tests_per_individual = " + str(tests_per_individual))

    # out degree of the vertices
    indeg = np.zeros(N + m)
    indeg[0:N] = tests_per_individual

    # in degree of the vertices
    outdeg = np.zeros(N+m)
    outdeg[N:N+m] = group_size

    # output the sum of indeg and outdeg if checking conditions
    if opts['verbose']:
        print("before fixing")
        print("outdeg = {}".format(np.sum(outdeg)))
        print("indeg = {}".format(np.sum(indeg)))

    # check if the number of tests per individual is less than the max, and, 
    # if we can, fix it
    if tests_per_individual < max_tests_per_individual:
        while(np.sum(indeg) < np.sum(outdeg)):
            index = np.random.randint(0,N)
            indeg[index] = indeg[index]+1
    else:
        try:
            assert np.sum(outdeg) == np.sum(indeg)
        except AssertionError:
            errstr = ("Assertion Failed: Require sum(outdeg) = " + str(np.sum(outdeg)) + " == " \
                + str(np.sum(indeg)) + " = sum(indeg)")
            print(errstr)
            print("out degree sequence: {}".format(outdeg.tolist()))
            print("in degree sequence: {}".format(indeg.tolist()))
            sys.exit()

    # output stats after fixing
    if opts['verbose']:
        print("after fixing")
        print("outdeg = {}".format(np.sum(outdeg)))
        print("indeg = {}".format(np.sum(indeg)))

    # generate the graph
    try:
        g = igraph.Graph.Degree_Sequence(outdeg.tolist(),indeg.tolist(),opts['graph_gen_method']) # options are "no_multiple" or "simple"
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
        if opts['verbose']:
            print(g)
            print(A)
            print("before resizing")
            print(A.shape)
            print("row sum {}".format(np.sum(A,axis=1)))
            print("column sum {}".format(np.sum(A,axis=0)))

        # the generated matrix has nonzeros in bottom left with zeros 
        # everywhere else, resize to it's m x N
        A = A[N:m+N,0:N]

        # check if the graph corresponds to a bipartite graph
        check_bipartite = g.is_bipartite()

        # save the row and column sums
        row_sum = np.sum(A,axis=1)
        col_sum = np.sum(A,axis=0)

        # display properties of A and graph g
        if opts['verbose']:
            print(A)
            print("after resizing")
            print(A.shape)
            print("row sum {}".format(row_sum))
            print("column sum {}".format(col_sum))
            print("max row sum {}".format(max(row_sum)))
            print("max column sum {}".format(max(col_sum)))
            print("min row sum {}".format(min(row_sum)))
            print("min column sum {}".format(min(col_sum)))
            print("g is bipartite: {}".format(check_bipartite))

        # set options and plot corresponding graph
        if opts['plotting']:
            layout = g.layout("auto")
            visual_style = {}
            visual_style['vertex_size'] = 10
            visual_style['layout'] = layout
            visual_style['edge_width'] = 0.2
            visual_style['edge_arrow_width'] = 0.1
            visual_style['bbox'] = (1200, 1200)
            igraph.drawing.plot(g, **visual_style)

        # save data to a MATLAB ".mat" file
        if opts['saving']:
            data = {}
            data['A'] = A
            data['bipartite'] = check_bipartite
            data['indeg'] = indeg
            data['outdeg'] = outdeg
            data['min_col_sum'] = min(col_sum)
            data['min_row_sum'] = min(row_sum)
            data['max_col_sum'] = max(col_sum)
            data['max_row_sum'] = max(row_sum)
            data['opts'] = opts
            data['graph_gen_method'] = opts['graph_gen_method']
            data['seed'] = opts['seed']
            sio.savemat('./' + opts['run_ID'] + '_generate_groups_output.mat', data)

    # return the adjacency matrix of the graph
    return A

# main method for testing
if __name__ == '__main__': 

    # print igraph version
    print("Loaded igraph version {}".format(igraph.__version__))

    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['m'] = 300
    opts['N'] = 600
    opts['group_size'] = 30
    opts['max_tests_per_individual'] = 15
    opts['graph_gen_method'] = 'simple'
    opts['verbose'] = False
    opts['plotting'] = False
    opts['saving'] = True
    opts['run_ID'] = 'GT_matrix_generation_component'
    opts['seed'] = 0

    # generate the measurement matrix with igraph
    A = gen_measurement_matrix(opts)

    # print shape of matrix
    if opts['verbose']:
        print("Generated adjacency matrix of size:")
        print(A.shape)
