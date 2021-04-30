import pulp as pl
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score

from group_testing.generate_test_results import gen_test_vector
import group_testing.utils as utils


class GroupTestingDecoder(BaseEstimator, ClassifierMixin):
    """
    A class to represent a decoder
    """
    def __init__(self, lambda_w=1, lambda_p=1, lambda_n=1, lambda_e=None, defective_num_lower_bound=None,
                 lp_relaxation=False, lp_rounding_threshold=0,
                 is_it_noiseless=True, solver_name=None, solver_options=None):
        """
        Constructs all the necessary attributes for the decoder object.

        Parameters:

            lambda_w (vector): A vector to provide prior weight. Default to vector of ones.
            lambda_p (int): Regularization coefficient for positive labels. Default to 1.
            lambda_n (int): Regularization coefficient for negative labels. Default to 1.
            lambda_e (int): Regularization coefficient for both negative and positive labels. Default to None. When is
            not None both lambda_p and lambda_n would be equal to each other and equal to value of lambda_e
            (i.e. lambda_p = lambda_n = lambda_e)
            defective_num_lower_bound (int): lower bound for number of infected people. Default to None.
            lp_relaxation (bool): A flag to use the lp relaxed version. Default to False.
            lp_rounding_threshold (float): Rounding to 0 and 1 for threshold for lp solutions. Default to 0.
            Range from 0 to 1.
            is_it_noiseless (bool): A flag to specify whether the problem is noisy or noiseless. Default to True.
            solver_name (str): Solver's name provided by Pulp. Default to None.
            solver_options (dic): Solver's options provided by Pulp. Default to None.

        """

        self.lambda_e = lambda_e
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n
        try:
            assert isinstance(lambda_w, (int, float, list))
            self.lambda_w = lambda_w
        except AssertionError:
            print("lambda_w should be either int, float or list of numbers")
        # -----------------------------------------
        self.defective_num_lower_bound = defective_num_lower_bound
        self.lp_relaxation = lp_relaxation
        self.lp_rounding_threshold = lp_rounding_threshold
        self.is_it_noiseless = is_it_noiseless
        self.solver_name = solver_name
        self.solver_options = solver_options
        self.prob_ = None
        # self.ep_cat = 'Binary'
        # self.en_cat = 'Binary'
        if self.lp_relaxation:
            self.en_upBound = 1
        else:
            self.en_upBound = None

    def fit(self, A, label):
        """
        Function to a decode base of design matrix and test results

        Parameters:

            A (binary numpy 2d-array): The group testing matrix.
            label (binary numpy array): The vector of results of the group tests.

        Returns:

            self (GroupTestingDecoder): A decoder object including decoding solution
        """
        if self.lambda_e is not None:
            # Use lambda_e if both lambda_p and lambda_n have same value
            self.lambda_p = self.lambda_e
            self.lambda_n = self.lambda_e
        m, n = A.shape
        alpha = A.sum(axis=1)
        label = np.array(label)
        positive_label = np.where(label == 1)[0]
        negative_label = np.where(label == 0)[0]
        # positive_label = [idx for idx,i in enumerate(label) if i==1]
        # negative_label = [idx for idx,i in enumerate(label) if i==0]
        # -------------------------------------
        # Checking length of lambda_w
        try:
            if isinstance(self.lambda_w, list):
                assert len(self.lambda_w) == n
        except AssertionError:
            print("length of lambda_w should be equal to number of individuals( numbers of columns in the group "
                  "testing matrix)")
        # -------------------------------------
        # Initializing the ILP problem
        p = pl.LpProblem('GroupTesting', pl.LpMinimize)
        # p.verbose(param['verbose'])
        # Variables kind
        if self.lp_relaxation:
            varCategory = 'Continuous'
            #self.solver_options['mip']= True
        else:
            varCategory = 'Binary'
        # Variable w
        w = pl.LpVariable.dicts('w', range(n), lowBound=0, upBound=1, cat=varCategory)
        # --------------------------------------
        # Noiseless setting
        if self.is_it_noiseless:
            # Defining the objective function
            p += pl.lpSum([self.lambda_w * w[i] if isinstance(self.lambda_w, (int, float)) else self.lambda_w[i] * w[i]
                        for i in range(n)])
            # Constraints
            for i in positive_label:
                p += pl.lpSum([A[i][j] * w[j] for j in range(n)]) >= 1
            for i in negative_label:
                p += pl.lpSum([A[i][j] * w[j] for j in range(n)]) == 0
            # Prevalence lower-bound
            if self.defective_num_lower_bound is not None:
                p += pl.lpSum([w[k] for k in range(n)]) >= self.defective_num_lower_bound

        # --------------------------------------
        # Noisy setting
        else:
            ep = []
            en = []
            # Variable ep
            if len(positive_label) != 0:
                ep = pl.LpVariable.dicts(name='ep', indexs=list(positive_label), lowBound=0, upBound=1, cat=varCategory)
            # Variable en
            if len(negative_label) != 0:
                en = pl.LpVariable.dicts(name='en', indexs=list(negative_label), lowBound=0, upBound=self.en_upBound,
                                      cat=varCategory)
            # Defining the objective function
            p += pl.lpSum([self.lambda_w * w[i] if isinstance(self.lambda_w, (int, float)) else self.lambda_w[i] * w[i]
                        for i in range(n)]) + \
                 pl.lpSum([self.lambda_p * ep[j] for j in positive_label]) + \
                 pl.lpSum([self.lambda_n * en[k] for k in negative_label])
            # Constraints
            for i in positive_label:
                p += pl.lpSum([A[i][j] * w[j] for j in range(n)] + ep[i]) >= 1
            for i in negative_label:
                if varCategory == 'Continuous':
                    p += pl.lpSum([A[i][j] * w[j] for j in range(n)] + -1 * en[i]) == 0
                else:
                    p += pl.lpSum([-1 * A[i][j] * w[j] for j in range(n)] + alpha[i] * en[i]) >= 0
            # Prevalence lower-bound
            if self.defective_num_lower_bound is not None:
                p += pl.lpSum([w[i] for i in range(n)]) >= self.defective_num_lower_bound
        if self.solver_options is not None:
            solver = pl.get_solver(self.solver_name, **self.solver_options)
        else:
            solver = pl.get_solver(self.solver_name)
        p.solve(solver)
        if not self.lp_relaxation:
            p.roundSolution()
        # ----------------
        self.prob_ = p
        #print("Status:", pl.LpStatus[p.status])
        return self

    def get_params_w(self, deep=True):
        """
        Function to provide a dictionary of individuals with their status obtained by decoder.

        Parameters:

            self (GroupTestingDecoder): Decoder object.

        Returns:

            w_solutions_dict (dict): A dictionary of individuals with their status.
        """
        variable_type = 'w'
        try:
            assert self.prob_ is not None
            # w_solution_dict = dict([(v.name, v.varValue)
            #                         for v in self.prob_.variables() if variable_type in v.name and v.varValue > 0])
            # TODO: Pulp uses ASCII sort when we recover the solution. It would cause a lot of problems when we want
            # TODO: to use the solution. We need to use alphabetical sort based on variables names (v.names). To do so
            # TODO: we use utils.py and the following lines of codes
            w_solution_dict = dict([(v.name, v.varValue)
                                    for v in self.prob_.variables() if variable_type in v.name])
            index_map = {v: i for i, v in enumerate(sorted(w_solution_dict.keys(), key=utils.natural_keys))}
            w_solution_dict = {k: v for k, v in sorted(w_solution_dict.items(), key=lambda pair: index_map[pair[0]])}
        except AttributeError:
            raise RuntimeError("You must fit the data first!")
        return w_solution_dict

    def solution(self):
        """
        Function to provide a vector of decoder solution.

        Parameters:

            self (GroupTestingDecoder): Decoder object.

        Returns:

            w_solutions (vector): A vector of decoder solution.
        """
        try:
            assert self.prob_ is not None
            # w_solution = [v.name[2:] for v in self.prob_.variables() if v.name[0] == 'w' and v.varValue > 0]
            # TODO: Pulp uses ASCII sort when we recover the solution. It would cause a lot of problems when we want
            # TODO: to use the solution. We need to use alphabetical sort based on variables names (v.names). To do so
            # TODO: we use utils.py and the following lines of codes
            w_solution = self.get_params_w()
            index_map = {v: i for i, v in enumerate(sorted(w_solution.keys(), key=utils.natural_keys))}
            w_solution = [v for k, v in sorted(w_solution.items(), key=lambda pair: index_map[pair[0]])]
            if self.lp_relaxation:
                w_solution = [1 if i > self.lp_rounding_threshold else 0 for i in w_solution]
        except AttributeError:
            raise RuntimeError("You must fit the data first!")
        return w_solution

    def predict(self, A):
        """
        Function to predict test results based on solution.

        Parameters:

            self (GroupTestingDecoder): Decoder object.
            A (binary numpy 2d-array): The group testing matrix.

        Returns:

             A vector of predicted test results based on the decoder solution.
        """
        return np.minimum(np.matmul(A, self.solution()), 1)

    def decodingScore(self, w_true):
        """
        Function to evaluate decoder's solution based on balanced_accuracy

        Parameters:

            self (GroupTestingDecoder): Decoder object.
            w_true (vector): True individual status value.

        Returns:

             Balanced accuracy of the decoder solution
        """
        return balanced_accuracy_score(w_true, self.solution())

    def write(self):
        pass
