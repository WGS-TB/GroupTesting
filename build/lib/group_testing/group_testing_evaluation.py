import sklearn
from sklearn.metrics import confusion_matrix
import os


def decoder_evaluation(w_true, sln, ev_metric='balanced_accuracy'):
    tn, fp, fn, tp = confusion_matrix(w_true, sln).ravel()
    eval_metric = getattr(sklearn.metrics,'{}_score'.format(ev_metric))
    eval_score = eval_metric(w_true, sln)
    ev_result = {'tn': tn, 'fp': fp, 'fn': fn, 'tp':tp, ev_metric:round(eval_score, 3)}
    print(ev_result)
    return ev_result


if __name__ == '__main__':
    # TODO: update this part
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, r'problem.mps')
    u = [1, 0, 0, 1, 0, 1]
    sln = {'w': [0, 3]}
    n = 6
    decoder_evaluation(u, sln, n)
