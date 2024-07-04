import numpy as np
from logistic_regression.stable_logistic import stable_sigmoid, stable_loss


def normallogistic(w, Master,yTr, X):
    """
    :param w: model
    :param Master: has coded data
    :param yTr: labels in +-1
    :param X: raw data
    :return: loss, gradient
    """
    dictionary = {}  # positional arg for master.query
    y_pred = Master.query(w, X, dictionary)
    vals = yTr * y_pred
    loss = stable_loss(-vals)
    func = np.vectorize(stable_sigmoid)
    den = func(-vals)  # computes sigmoid func 1/1+e^{-x}, so pass -vals for correct gradient
    alpha = yTr * den
    gradient = - Master.query(alpha.reshape(1, -1), X, dictionary).reshape(-1, 1)/len(yTr)
    return loss, gradient
