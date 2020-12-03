from pymprog import model
import os
import numpy as np
from generate_test_results import gen_test_vector


def problem_setup(A, label, param):
    # This function provide a ILP in form of eq 11 in WABI paper. However it can be changed if we uncomment
    # "Additional constraints" below.
    file_path = param['file_path']
    lambda_w = param['lambda_w']
    lambda_p = param['lambda_p']
    lambda_n = param['lambda_n']
    fixed_defective_num = param['defective_num']
    sensitivity = param['sensitivity']
    specificity = param['specificity']
    noiseless_mode = param['noiseless_mode']
    LP_relaxation = param['LP_relaxation']
    m, n = A.shape
    alpha = A.sum(axis=1)
    # label = np.array(label)
    positive_label = np.where(label == 1)[0]
    negative_label = np.where(label == 0)[0]

    # We are using pymprog
    # Initializing the ILP problem
    p = model('GroupTesting')
    p.verbose(param['verbose'])
    # Variables kind
    if LP_relaxation:
        varKind = float
    else:
        varKind = bool
    # Variable w
    #TODO: Fix w lowerbound and upperbound
    w = p.var(name='w', inds=n, bounds=(0, 1), kind=varKind)
    # Variable ep
    if len(positive_label)!=0:
        ep = p.var(name='ep', inds=list(positive_label), kind=float, bounds=(0, 1))
    # Variable en
    if len(negative_label)!=0:
        if noiseless_mode:
            en = p.var(name='en', inds=list(negative_label))
        else:
            en = p.var(name='en', inds=list(negative_label), kind=bool)
    # Defining the objective function
    prob_obj = sum(lambda_w * w[i] for i in range(n)) + \
               sum(lambda_p * ep[j] for j in positive_label) + \
               sum(lambda_n * en[k] for k in negative_label)

    p.minimize(prob_obj, name='OBJ')
    # Constraints
    for i in positive_label:
        sum(A[i][j] * w[j] for j in range(n)) + ep[i] >= 1
    for i in negative_label:
        if noiseless_mode:
            sum(A[i][j] * w[j] for j in range(n)) - en[i] == 0
        else:
            sum(-1 * A[i][j] * w[j] for j in range(n)) + alpha[i] * en[i] >= 0

    # Additional constraints

    # if fixed_defective_num is not None:
    #     sum(w[i] for i in range(n)) <= fixed_defective_num

    # if sensitivity is not None:
    #     sum(en[i] for i in negative_label) <= sensitivity* len(Z)

    # TODO: Right now we need to solve the problem before saving the file! Otherwise it wouldn't save the constraints.
    # TODO: For now we can set a time limit
    p.solver(int, tm_lim=1)
    p.solve()
    # Save fixed mps format
    p.write_mps(1, None, file_path)
    # Change the line above to p.write_mps(2, None, file_path) if you want free mps format.
    p.end()

if __name__ == '__main__':
    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['m'] = 3
    opts['N'] = 5
    opts['verbose'] = True  # False
    opts['plotting'] = True  # False
    opts['saving'] = True
    opts['run_ID'] = 'GT_test_result_vector_generation_component'
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'
    opts['test_noise_methods'] = ['truncation']
    opts['seed'] = 0

    A = np.random.randint(2, size=(opts['m'], opts['N']))
    u = np.random.randint(2, size=opts['N'])
    b = gen_test_vector(A, u, opts)

    # Test
    #A = np.array([[1,0,0],[1,0,1],[0,1,0]])
    #b = np.array([1,0,1])
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, r'problem.mps')

    param = {}
    param['file_path'] = file_path
    param['lambda_w'] = 1
    param['lambda_p'] = 100
    param['lambda_n'] = 100
    param['verbose'] = False
    param['defective_num'] = None
    param['sensitivity'] = None
    param['specificity'] = None
    param['noiseless_mode'] = True
    param['LP_relaxation'] = False

    problem_setup(A, b, param)
