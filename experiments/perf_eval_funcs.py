import numpy as np
from sklearn.metrics import log_loss,accuracy_score

def binary_logistic_loss(y_pred,y,**kwargs):    
    return log_loss(y,y_pred)

def multiclass_logistic_loss(y_pred, y, **kwargs):
    """Calculate average logistic loss
    over all data points for multi-class classification

    :return: logistic loss
    :rtype: float
    """
    # In the multi-class setting, y_pred is an i x k matrix
    # where i is the number of samples and k is the number of classes
    # Each entry is the probability of predicting the kth class
    # for the ith sample. We need to get the probability of predicting
    # the true class for each sample and then take the sum of the
    # logs of that.
    n = len(y)
    probs_trueclasses = y_pred[np.arange(n), y.astype("int")]
    return -1 / n * sum(np.log(probs_trueclasses))

def probabilistic_accuracy(y_pred, y, **kwargs):
    """For binary classification only.
    1 - error rate. Use when output of 
    model y_pred is a probability

    :param y_pred: Array of predicted probabilities of each label
    :param y: Array of true labels, 1-dimensional

    """
    v = np.where(y != 1.0, 1.0 - y_pred, y_pred)
    return sum(v) / len(v)

def multiclass_accuracy(y_pred,y,**kwargs):
    """For multi-class classification.
    1 - error rate. Use when output of 
    model y_pred is a probability

    :param y_pred: Array of predicted probabilities of each label
    :param y: Array of true labels, 1-dimensional

    """
    n = len(y)
    return np.sum(y_pred[np.arange(n),y.astype("int")])/n

def deterministic_accuracy(y_pred, y, **kwargs):
    """The fraction of correct samples. Best to use
    only when the output of the model, y_pred
    is 0 or 1. 

    :param y_pred: Array of predicted labels
    :param y: Array of true labels

    """
    from sklearn.metrics import accuracy_score
    return accuracy_score(y,y_pred > 0.5)


def MSE(y_pred, y, **kwargs):
    """Calculate sample mean squared error

    :param y_pred: Array of predicted labels
    :param y: Array of true labels
    """
    n = len(y)
    res = sum(pow(y_pred - y, 2)) / n
    return res