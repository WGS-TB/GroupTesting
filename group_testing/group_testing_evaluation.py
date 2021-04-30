import sklearn
from sklearn.metrics import confusion_matrix
import os


def decoder_evaluation(u_true, sln, ev_metric='balanced_accuracy'):
    """
    Function to evaluate decoder's solution

    Parameters:
        u_true (vector): True individual status value.
        sln (vector): Predicted individual status value.
        ev_metric (str): evaluation metric. Default to 'balanced_accuracy'.

    Returns:
        ev_result (dictionary): tn, fp, fn, tp, value of selected evaluation metric.
    """
    tn, fp, fn, tp = confusion_matrix(u_true, sln).ravel()
    eval_metric = getattr(sklearn.metrics,'{}_score'.format(ev_metric))
    eval_score = eval_metric(u_true, sln)
    ev_result = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, ev_metric: round(eval_score, 3)}
    return ev_result


if __name__ == '__main__':
    """
    Main method for testing decoder_evaluation
    """
    u_true = [1, 0, 0, 1, 0, 1]
    sln = [1, 1, 0, 0, 0, 1]
    decoder_evaluation(u_true=u_true, sln=sln,ev_metric='balanced_accuracy')
