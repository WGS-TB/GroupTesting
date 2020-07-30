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

"""
Unit testing to be added later
"""
#class TestMeasurementMatrix(unittest.TestCase):
#
#    def test_make_graph(self):
#        gen_measurement_matrix(opts)
        

"""
Function to generate and return the matrix
"""
def gen_measurement_matrix(opts):

    # set the seed used for graph generation to the options seed
    random.seed(opts['seed'])
    np.random.seed(opts['seed'])

    # (slightly) shorter references for options
    m = opts['m']
    N = opts['N']
    group_size = opts['group_size']
    max_tests_per_individual = opts['max_tests_per_individual']

    # compute tests per individual needed to satisfy 
    #           N*tests_per_individual == m*group_size
    avg_test_split = math.floor(m*group_size/N)

    # verify the given parameters can lead to a valid testing scheme:
    #   if floor(m*group_size/N) > max_tests_per_individual, we would need to 
    #   test each individual more times than the maximum allowed to satisfy 
    #   the group size constraint
    try:
        assert avg_test_split <= max_tests_per_individual
        assert m <= math.floor(N/2)
    except AssertionError:
        errstr = ("Assertion Failed: With group_size = " + str(group_size) \
            + ", since m = " + str(m) + " and N = " + str(N) \
            + " we have floor(m*group_size/N) = " + str(avg_test_split) \
            + " which exceeds max_tests_per_individual = " + str(max_tests_per_individual)) 
        print(errstr)
        sys.exit()

    # compute the actual tests per individual
    # note: we may end up with individuals being tested less than the maximum
    #   allowed number of tests, but this is not a problem (only the opposite is)
    tests_per_individual = max(min(max_tests_per_individual, avg_test_split),1)

    if opts['verbose']:
        print("tests_per_individual = " + str(tests_per_individual))

    # In this model, the first N elements of the degree sequence correspond
    # to the population, while the last m elements correspond to the groups

    # in-degree corresponds to the individuals, here we set the first N 
    # entries of the in-degree sequence to specify the individuals 
    indeg = np.zeros(N + m)
    indeg[0:N] = tests_per_individual

    # out-degree corresponds to the group tests, here we set the last m 
    # entries of the out-degree sequence to specify the groups
    outdeg = np.zeros(N+m)
    outdeg[N:N+m] = group_size

    # keep track of vertex types for final graph vertex coloring
    vtypes = np.zeros(N + m)
    vtypes[0:N] = 1

    # output the sum of indeg and outdeg if checking conditions
    if opts['verbose']:
        print("out degree sequence: {}".format(outdeg.tolist()))
        print("in degree sequence:  {}".format(indeg.tolist()))
        print("sum outdeg (groups) = {}".format(np.sum(outdeg)))
        print("sum indeg (individ) = {}".format(np.sum(indeg)))

    # check if the number of tests per individual is less than the max, and, 
    # if we can, fix it
    if tests_per_individual < max_tests_per_individual:
        if (np.sum(indeg) < np.sum(outdeg)):
            while(np.sum(indeg) < np.sum(outdeg)):
                # select index randomly (bad: might cause largest index to be updated, 
                #   possibly exceed max_tests_per_individual)
                #index = np.random.randint(0,N)

                # select index corresponding to one of the minimum indices
                nz_indeg = indeg[(indeg > 0)]
                min_indices = np.array(np.where(indeg == nz_indeg.min()))
                #min_indices = indeg[(indeg == nz_indeg.min())]# & (indeg > 0)]
                #min_indices = np.array(np.where((indeg == indeg.min()) & (indeg > 0)))
                #print(min_indices)
                #print(min_indices.shape[1])
                index = np.random.randint(min_indices.shape[1], size=1)
                #print('generated index = ' + str(index))
                indeg[min_indices[0,index]] = indeg[min_indices[0,index]]+1
        
            # output stats after fixing
            if opts['verbose']:
                print("after fixing")
                print("out degree sequence: {}".format(outdeg.tolist()))
                print("in degree sequence:  {}".format(indeg.tolist()))
                print("sum outdeg (groups) = {}".format(np.sum(outdeg)))
                print("sum indeg (individ) = {}".format(np.sum(indeg)))

    else:
        try:
            assert np.sum(outdeg) == np.sum(indeg)
        except AssertionError:
            errstr = ("Assertion Failed: Require sum(outdeg) = " + str(np.sum(outdeg)) + " == " \
                + str(np.sum(indeg)) + " = sum(indeg)")
            print(errstr)
            print("out degree sequence: {}".format(outdeg.tolist()))
            print("in degree sequence: {}".format(indeg.tolist()))
            #sys.exit()

    # generate the graph
    try:
        g = igraph.Graph.Degree_Sequence(outdeg.tolist(),indeg.tolist(),opts['graph_gen_method']) 
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
        if opts['verbose']:
            #print(g)
            #print(A)
            #print("before resizing")
            #print(A.shape)
            print("row sum {}".format(np.sum(A,axis=1)))
            print("column sum {}".format(np.sum(A,axis=0)))

        # the generated matrix has nonzeros in bottom left with zeros 
        # everywhere else, resize to it's m x N
        A = A[N:m+N,0:N]
        #A = np.minimum(A, 1)

        # check if the graph corresponds to a bipartite graph
        check_bipartite = g.is_bipartite()

        # save the row and column sums
        row_sum = np.sum(A,axis=1)
        col_sum = np.sum(A,axis=0)

        # display properties of A and graph g
        if opts['verbose']:
            #print(A)
            #print("after resizing")
            #print(A.shape)
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
            color_dict = {1: "blue", 0: "red"}
            g.vs['color'] = [color_dict[vertex_type] for vertex_type in g.vs['vertex_type']]
            B = g.vs.select(vertex_type = 'B')
            C = g.vs.select(vertex_type = 'C')
            visual_style = {}
            visual_style['vertex_size'] = 10
            visual_style['layout'] = layout
            visual_style['edge_width'] = 0.5
            visual_style['edge_arrow_width'] = 0.2
            visual_style['bbox'] = (1200, 1200)
            igraph.drawing.plot(g, **visual_style)

        # save data to a MATLAB ".mat" file
        # if opts['saving']:
        #     data = {}
        #     data['A'] = A
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

# main method for testing
if __name__ == '__main__': 

    # print igraph version
    print("Loaded igraph version {}".format(igraph.__version__))

    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['m'] = 100
    opts['N'] = 500
    opts['group_size'] = 30
    opts['max_tests_per_individual'] = 15
    opts['graph_gen_method'] = 'no_multiple' # options are "no_multiple" or "simple"
    opts['verbose'] = True #False
    opts['plotting'] = True #False
    opts['saving'] = True
    opts['run_ID'] = 'GT_matrix_generation_component'
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'
    opts['seed'] = 0

    # generate the measurement matrix with igraph
    A = gen_measurement_matrix(opts)

    # print shape of matrix
    if opts['verbose']:
        print("Generated adjacency matrix of size:")
        print(A.shape)
