import pulp as pl
from pulp import *
import os
import numpy as np
from generate_test_results import gen_test_vector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from utils import *

class GroupTestingDecoder(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda_w=1, lambda_p=1, lambda_n=1, lambda_e=None, defective_num_lower_bound=None, sensitivity_threshold=None,
                 specificity_threshold=None, lp_relaxation=False, is_it_noiseless=True, solver_name=None, solver_options=None):
        # TODO: Check their values
        # TODO: Change lambda_w to sample weight
        if lambda_e is not None:
            # Use lambda_e if both lambda_p and lambda_n have same value
            self.lambda_p = lambda_e
            self.lambda_n = lambda_e
            print('single lambda!')
        else:
            self.lambda_p = lambda_p
            self.lambda_n = lambda_n
            print('two lambdas!')
        # -----------------------------------------
        # lambda_w is added as a coefficient for vector w. lambda_w could be used as a vector of prior probabilities.
        # lambda_w default value is 1.
        try:
            assert isinstance(lambda_w, (int, float, list))
            self.lambda_w = lambda_w
        except AssertionError:
            print("lambda_w should be either int, float or list of numbers")
        # -----------------------------------------
        self.defective_num_lower_bound = defective_num_lower_bound
        self.sensitivity_threshold = sensitivity_threshold
        self.specificity_threshold = specificity_threshold
        self.lp_relaxation = lp_relaxation
        self.is_it_noiseless = is_it_noiseless  # TODO: Do we need this?
        self.solver_name = solver_name
        self.solver_options = solver_options
        self.prob_ = None
        self.ep_cat = 'Binary'
        self.en_cat = 'Binary'
        self.en_upBound = 1

    def fit(self, A, label):
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
        p = LpProblem('GroupTesting', LpMinimize)
        # p.verbose(param['verbose'])
        # Variables kind
        if self.lp_relaxation:
            varCategory = 'Continuous'
        else:
            varCategory = 'Binary'
        # Variable w
        w = LpVariable.dicts('w', range(n), lowBound=0, upBound=1, cat=varCategory)
        # --------------------------------------
        # Noiseless setting
        if self.is_it_noiseless:
            # Defining the objective function
            p += lpSum([self.lambda_w * w[i] if isinstance(self.lambda_w, (int, float)) else self.lambda_w[i] * w[i]
                        for i in range(n)])
            # Constraints
            for i in positive_label:
                p += lpSum([A[i][j] * w[j] for j in range(n)]) >= 1
            for i in negative_label:
                p += lpSum([A[i][j] * w[j] for j in range(n)]) == 0
            # Prevalence lower-bound
            if self.defective_num_lower_bound is not None:
                p += lpSum([w[k] for k in range(n)]) >= self.defective_num_lower_bound
            print(p)

        # --------------------------------------
        # Noisy setting
        else:
            ep = []
            en = []
            # Variable ep
            if len(positive_label) != 0:
                ep = LpVariable.dicts(name='ep', indexs=list(positive_label), lowBound=0, upBound=1, cat=self.ep_cat)
            # Variable en
            if len(negative_label) != 0:
                en = LpVariable.dicts(name='en', indexs=list(negative_label), lowBound=0, upBound=self.en_upBound,
                                      cat=self.en_cat)
            # Defining the objective function
            p += lpSum([self.lambda_w * w[i] if isinstance(self.lambda_w, (int, float)) else self.lambda_w[i] * w[i]
                        for i in range(n)]) + \
                 lpSum([self.lambda_p * ep[j] for j in positive_label]) + \
                 lpSum([self.lambda_n * en[k] for k in negative_label])
            # TODO: what if we call ep and en without defining them? above
            # Constraints
            for i in positive_label:
                p += lpSum([A[i][j] * w[j] for j in range(n)] + ep[i]) >= 1
            for i in negative_label:
                if self.en_cat == 'Continuous':
                    p += lpSum([A[i][j] * w[j] for j in range(n)] + -1 * en[i]) == 0
                else:
                    p += lpSum([-1 * A[i][j] * w[j] for j in range(n)] + alpha[i] * en[i]) >= 0
            # TODO: add additional constraints
            # Prevalence lower-bound
            if self.defective_num_lower_bound is not None:
                p += lpSum([w[i] for i in range(n)]) >= self.defective_num_lower_bound
        solver = pl.get_solver(self.solver_name, **self.solver_options)
        p.solve(solver)
        # TODO: Check this
        p.roundSolution()
        # ----------------
        self.prob_ = p
        print("Status:", LpStatus[p.status])
        return self

    def get_params(self, variable_type='w'):
        try:
            assert self.prob_ is not None
            # w_solution_dict = dict([(v.name, v.varValue)
            #                         for v in self.prob_.variables() if variable_type in v.name and v.varValue > 0])
            # TODO: Pulp uses ASCII sort when we recover the solution. It would cause a lot of problems when we want
            # TODO: to use the solution. We need to use alphabetical sort based on variables names (v.names). To do so
            # TODO: we use utils.py and the following lines of codes
            w_solution_dict = dict([(v.name, v.varValue)
                                    for v in self.prob_.variables() if variable_type in v.name])
            index_map = {v: i for i, v in enumerate(sorted(w_solution_dict.keys(), key=natural_keys))}
            w_solution_dict = {k: v for k, v in sorted(w_solution_dict.items(), key=lambda pair: index_map[pair[0]])}
        except AttributeError:
            raise RuntimeError("You must fit the data first!")
        return w_solution_dict

    def solution(self):
        try:
            assert self.prob_ is not None
            # w_solution = [v.name[2:] for v in self.prob_.variables() if v.name[0] == 'w' and v.varValue > 0]
            # TODO: Pulp uses ASCII sort when we recover the solution. It would cause a lot of problems when we want
            # TODO: to use the solution. We need to use alphabetical sort based on variables names (v.names). To do so
            # TODO: we use utils.py and the following lines of codes
            w_solution = self.get_params(variable_type='w')
            index_map = {v: i for i, v in enumerate(sorted(w_solution.keys(), key=natural_keys))}
            w_solution = [v for k, v in sorted(w_solution.items(), key=lambda pair: index_map[pair[0]])]
        except AttributeError:
            raise RuntimeError("You must fit the data first!")
        return w_solution

    def predict(self, X):
        return np.minimum(np.matmul(X, self.solution()), 1)

    # def score(self):
    #     pass

    def decodingScore(self, w_true):
        return balanced_accuracy_score(w_true, self.solution())

    def write(self):
        pass


if __name__ == '__main__':
    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['m'] = 150
    opts['N'] = 300
    opts['verbose'] = True  # False
    opts['plotting'] = True  # False
    opts['saving'] = True
    opts['run_ID'] = 'GT_test_result_vector_generation_component'
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'
    opts['test_noise_methods'] = []
    opts['seed'] = 0

    A = np.random.randint(2, size=(opts['m'], opts['N']))
    u = np.random.randint(2, size=opts['N'])
    b = gen_test_vector(A, u, opts)
    # print(A)
    # print(b)
    # print(u)

    # Test
    # A = np.array([[1,0,0],[1,0,1],[0,1,0]])
    # b = np.array([1,0,1])
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, r'problem.mps')

    param = {}
    # param['file_path'] = file_path
    param['lambda_w'] = 1
    param['lambda_p'] = 100
    param['lambda_n'] = 100
    # param['verbose'] = False
    param['fixed_defective_num'] = None
    param['sensitivity_threshold'] = None
    param['specificity_threshold'] = None
    param['is_it_noiseless'] = True
    param['lp_relaxation'] = False
    param['solver_name'] = 'COIN_CMD'
    param['solver_options'] = {'timeLimit': 60, 'logPath': 'log.txt'}

    c = GroupTestingDecoder(**param)
    c.fit(A, b)
    print(c.decodingScore(b))

