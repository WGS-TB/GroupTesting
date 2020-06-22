# import standard libraries
import time, os, argparse, io, shutil, sys, math, socket
import numpy as np
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
        

# reserved for later
#def gen_measurement_matrix(m, N, method="simple"):
#    return A

if __name__ == '__main__': 

    print("Loaded igraph version {}".format(igraph.__version__))

    opts = {}
    opts["verbose"] = True#False

    # maximum size of each group (#1s on each row)
    group_size = 2

    # maximum number of tests we can run per individual (#1s on each column)
    max_tests_per_individual = 1

    # number of tests
    m = 5

    # total population size
    N = 20

    try:
        assert m <= math.ceil(N/2)
    except AssertionError:
        errstr = ("Assertion Failed: With group_size = " + str(group_size) \
            + " and max_tests_per_individual = " + str(max_tests_per_individual) \
            + ", m = " + str(m) + " must be less than N/2 = " + str(math.ceil(N/2)) \
            + " (or else some individuals will be tested more than the max times)")
        print(errstr)
        #sys.exit()
        

    # out degree of the vertices
    outdeg = np.zeros(N)
    outdeg[0:m] = group_size

    # in degree of the vertices
    indeg = np.zeros(N)

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

    if opts["verbose"]:
        print("tests_per_individual = " + str(tests_per_individual))

    # 
    indeg[0:N] = tests_per_individual;

    if opts["verbose"]:
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
            sys.exit()

    if opts["verbose"]:
        print("after fixing")
        print("outdeg = {}".format(np.sum(outdeg)))
        print("indeg = {}".format(np.sum(indeg)))


    #outdeg = outdeg*3
    try:
        #g = igraph.Graph.Degree_Sequence(outdeg.tolist(),method="vl")
        g = igraph.Graph.Degree_Sequence(outdeg.tolist(),indeg.tolist(),method="no_multiple") # options are "no_multiple" or "simple"
        #g = igraph.Graph.Erdos_Renyi(10,m=5,directed=False,loops=False)
    except igraph._igraph.InternalError as err:
        print("igraph InternalError (likely invalid outdeg or indeg sequence): {0}".format(err))
        print("out degree sequence: {}".format(outdeg.tolist()))
        print("in degree sequence: {}".format(indeg.tolist()))
    except:
        print("Unexpected error:", sys.exec_info()[0])
    else:
        A = np.array(g.get_adjacency()._get_data())
        if opts["verbose"]:
            print(g)
            print(A)
            print("before resizing")
            print(A.shape)
            print("row sum {}".format(np.sum(A,axis=1)))
            print("column sum {}".format(np.sum(A,axis=0)))

        A = A[0:m,0:N]

        if opts["verbose"]:
            print(A)
            print("after resizing")
            print(A.shape)
            print("row sum {}".format(np.sum(A,axis=1)))
            print("column sum {}".format(np.sum(A,axis=0)))
            print("g is bipartite: {}".format(g.is_bipartite()))

        layout = g.layout("auto")
        visual_style = {}
        visual_style["vertex_size"] = 10
        visual_style["layout"] = layout
        visual_style["edge_width"] = 0.2
        visual_style["edge_arrow_width"] = 0.1
        visual_style["bbox"] = (1200, 1200)
        igraph.drawing.plot(g, **visual_style)

        data = {}
        data['A'] = A
        sio.savemat('./run_data', data)
