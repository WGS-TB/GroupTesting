from sklearn.metrics import *
import pandas as pd
import os


# def sln_indices(sln, key_value):
#     return sln[key_value]
#
#
# def sln_vector(sln, n):
#     sln_vec = [1 if i in sln else 0 for i in range(n)]
#     return sln_vec
#
#
# def decoder_evaluation(u, sln, n, ev_metric='all'):
#     #sln_vec = sln_vector(sln_indices(sln, "w"), n)
#     tn, fp, fn, tp = confusion_matrix(u, sln).ravel()
#     #acc = accuracy_score(u, sln_vec)
#     #TPR = recall_score(u, sln_vec)
#     ev_result = dict(tn=tn, fp=fp, fn=fn, tp=tp)
#     print(ev_result)
#     return ev_result

def decoder_evaluation(w_true, sln , ev_metric='all'):
    tn, fp, fn, tp = confusion_matrix(w_true, sln).ravel()
    balanced_accuracy = balanced_accuracy_score(w_true, sln)
    ev_result = dict(tn=tn, fp=fp, fn=fn, tp=tp, balanced_accuracy=round(balanced_accuracy,3))
    print(ev_result)
    return ev_result

if __name__ == '__main__':
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, r'problem.mps')
    u = [1, 0, 0, 1, 0, 1]
    sln = {'w': [0, 3]}
    n = 6
    decoder_evaluation(u, sln, n)
