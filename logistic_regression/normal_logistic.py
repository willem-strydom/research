
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
    dictionary = {}
    y_pred = Master.query(w, X, dictionary)
    vals = yTr * y_pred
    loss = stable_loss(-vals) #loss = np.sum(np.log(1 + np.exp(-yTr * y_pred)))
    # stable_sig = func(values)
    # normal_sig = 1 / (1 + np.exp(-values))
    func = np.vectorize(stable_sigmoid)
    den = func(-vals)
    # den0 = 1/(1 + np.exp(-vals)) (old way)
    alpha = yTr * den
    gradient = - Master.query(alpha.reshape(1, -1), X, dictionary).reshape(-1, 1)/len(yTr)
    return loss, gradient
