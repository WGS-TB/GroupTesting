#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hoomanzabeti
"""
import os
import numpy as np
from generate_test_results import gen_test_vector

def mps_writer(A,label,param):
    
    file_path = param['file_path']
    lambda_w = param['lambda_w']
    lambda_p = param['lambda_p']
    lambda_n = param['lambda_n']
    m,n= A.shape
    alpha = A.sum(axis=1)
    positive_label = np.where(label==1)[0]
    negative_label = np.where(label==0)[0]
    
    
    opt_file = open(file_path, 'w')
    opt_file.write('NAME\tProblem\n')
    opt_file.write('OBJSENSE\n')
    opt_file.write('\tMIN\n')
    opt_file.write('OBJNAME\n')
    opt_file.write('\tOBJ\n')
    
    # ---------------------- Print ROWS -------------------------
    opt_file.write('ROWS\n')
    opt_file.write(' N   OBJ\n')
    
    for i in range(m):
        if i in positive_label:
            opt_file.write(' G   cp{}\n'.format(i))
        elif i in negative_label:    
            opt_file.write(' G   cn{}\n'.format(i))
    
    # ---------------------- Print COLUMNS -------------------------
    opt_file.write('COLUMNS\n')
    w_column = ''
    for i in range(n): 
        w_column+='    w{}\tOBJ\t{}\n'.format(i,lambda_w)
        for j in np.where(A[:,i]==1)[0]:
            if j in positive_label: 
                w_column+='    w{}\tcp{}\t1\n'.format(i,j)
            elif j in negative_label:
                w_column+='    w{}\tcn{}\t-1\n'.format(i,j)
    opt_file.write(w_column)
            
    ep_column=''
    for i in positive_label: 
        ep_column+='    ep{}\tOBJ\t{}\n'.format(i,lambda_p)
        ep_column+='    ep{}\tcp{}\t1\n'.format(i,i)
    opt_file.write(ep_column)
    
    en_column=''
    for i in negative_label: 
        en_column+='    en{}\tOBJ\t{}\n'.format(i,lambda_n)
        en_column+='    en{}\tcn{}\t{}\n'.format(i,i,alpha[i])
    opt_file.write(en_column)
    
    # ---------------------- Print RHS -------------------------
    opt_file.write('RHS\n')
    rhs_columns = ''
    for i in positive_label: rhs_columns+='    RHS1\tcp{}\t1\n'.format(i)
    
    opt_file.write(rhs_columns)
    # ---------------------- Print BOUNDS -------------------------
    opt_file.write('BOUNDS\n')
    for i in range(n):
        opt_file.write('   BV BND\tw{}\n'.format(i))
#    for i in positive_label:
#        opt_file.write('   BV BND\tep{}\n'.format(i))
    # =============================================================================
    for i in positive_label:
         opt_file.write('   LO BND\tep{}\t0\n'.format(i))
         opt_file.write('   UP BND\tep{}\t1\n'.format(i))
    # =============================================================================
    for i in negative_label: 
        opt_file.write('   BV BND\ten{}\n'.format(i))
    opt_file.write('ENDATA')
    opt_file.close()
if __name__ == '__main__':

    # options for plotting, verbose output, saving, seed
    opts = {}
    opts['m'] = 4
    opts['N'] = 6
    opts['verbose'] = True #False
    opts['plotting'] = True #False
    opts['saving'] = True
    opts['run_ID'] = 'GT_test_result_vector_generation_component'
    opts['data_filename'] = opts['run_ID'] + '_generate_groups_output.mat'
    opts['seed'] = 0

    A = np.random.randint(2,size=(opts['m'],opts['N']))
    u = np.random.randint(2,size=opts['N'])
    b = gen_test_vector(A, u, opts)

    current_directory = os.getcwd()
    file_path = os.path.join(current_directory,r'problem.mps')
    
    param= {}
    param['file_path'] = file_path
    param['lambda_w'] = 1
    param['lambda_p'] = 0.1
    param['lambda_n'] = 0.2
    
    mps_writer(A,b,param)