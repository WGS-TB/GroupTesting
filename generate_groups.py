# import standard libraries
import time, os, argparse, io, shutil, sys, math, socket
import numpy as np
import random
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=60, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
#import hdf5storage
import matplotlib
import igraph
matplotlib.use('Agg')
import scipy.io as sio
#import unittest

# import plotting and data-manipulation tools
import matplotlib.pyplot as plt

#class TestMeasurementMatrix(unittest.TestCase):
#
#    def test_make_graph(self):
#        g = igraph.Graph.Degree_Sequence(outdeg.tolist(),indeg.tolist(),method="simple")
        

# function to generate and return the matrix
def gen_measurement_matrix(m, N, group_size = 30, max_tests_per_individual = 16, opts = {}, method="simple"):

    random.seed(opts['seed'])
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

    tests_per_individual = min(max_tests_per_individual, math.floor(m*group_size/N))

    if opts['verbose']:
        print("tests_per_individual = " + str(tests_per_individual))

    # out degree of the vertices
    indeg = np.zeros(N + m)
    indeg[0:N] = tests_per_individual

    # in degree of the vertices
    outdeg = np.zeros(N+m)
    outdeg[N:N+m] = group_size

    if opts['verbose']:
        print("before fixing")
        print("outdeg = {}".format(np.sum(outdeg)))
        print("indeg = {}".format(np.sum(indeg)))

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

    if opts['verbose']:
        print("after fixing")
        print("outdeg = {}".format(np.sum(outdeg)))
        print("indeg = {}".format(np.sum(indeg)))


    #outdeg = outdeg*3
    try:
        #g = igraph.Graph.Degree_Sequence(outdeg.tolist(),method="vl")
        g = igraph.Graph.Degree_Sequence(outdeg.tolist(),indeg.tolist(),method) # options are "no_multiple" or "simple"
        #g = igraph.Graph.Erdos_Renyi(10,m=5,directed=False,loops=False)
    except igraph._igraph.InternalError as err:
        print("igraph InternalError (likely invalid outdeg or indeg sequence): {0}".format(err))
        print("out degree sequence: {}".format(outdeg.tolist()))
        print("in degree sequence: {}".format(indeg.tolist()))
        sys.exit()
    except:
        print("Unexpected error:", sys.exec_info()[0])
    else:
        A = np.array(g.get_adjacency()._get_data())
        if opts['verbose']:
            print(g)
            print(A)
            print("before resizing")
            print(A.shape)
            print("row sum {}".format(np.sum(A,axis=1)))
            print("column sum {}".format(np.sum(A,axis=0)))

        A = A[N:m+N,0:N]

        check_bipartite = g.is_bipartite()
        row_sum = np.sum(A,axis=1)
        col_sum = np.sum(A,axis=0)

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

        if opts['plotting']:
            layout = g.layout("auto")
            visual_style = {}
            visual_style['vertex_size'] = 10
            visual_style['layout'] = layout
            visual_style['edge_width'] = 0.2
            visual_style['edge_arrow_width'] = 0.1
            visual_style['bbox'] = (1200, 1200)
            igraph.drawing.plot(g, **visual_style)

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
            data['m'] = m
            data['N'] = N
            data['group_size'] = group_size
            data['max_tests_per_individual'] = max_tests_per_individual
            data['opts'] = opts
            data['method'] = method
            data['seed'] = opts['seed']
            sio.savemat('./run_data.mat', data)

    return A

# main method for testing
if __name__ == '__main__': 

    print("Loaded igraph version {}".format(igraph.__version__))

    opts = {}
    opts['verbose'] = False
    opts['plotting'] = True
    opts['saving'] = True
    opts['seed'] = 0

    # maximum size of each group (#1s on each row)
    group_size = 30

    # maximum number of tests we can run per individual (#1s on each column)
    max_tests_per_individual = 15

    # number of tests
    m = 300

    # total population size
    N = 600

    A = gen_measurement_matrix(m, N, group_size, max_tests_per_individual, opts = opts, method="simple")

    if opts['verbose']:
        print("Generated adjacency matrix of size:")
        print(A.shape)
