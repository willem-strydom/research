import numpy as np
import pandas as pd
from quantization.quantize import quantize
from logistic_regression.stable_logistic import stable_sigmoid, stable_loss


def quant_logistic(w, Master, w_lvl, grd_lvl, dict, X, y, filename, index):
    """
    :param w: weights vector, needs to be numpy column vec
    :param Master: master instance
    :param w_lvl: quantization level for w
    :param grd_lvl: quantization level for grd
    :param dict: for performance recording
    :param X: raw data
    :param y: labels
    :param filename: where dict is recorded as csv
    :param index: for lookup table to make +-1 queries
    :return: gradient, as numpy column vector
    """
    y_pred = Master.uniform_query(w, w_lvl, dict, X, index)
    vals = y * y_pred
    loss = stable_loss(-vals)
    record_access(dict, filename)
    func = np.vectorize(stable_sigmoid)
    den = func(-vals)
    dict = {
        'w-quantization': [w_lvl],
        'grd-quantization': [grd_lvl],
        'imputation': [0],
        'access': 0,
        'query type': [0],
        'time': [0],
        'stop cond': [0],
        'iters': [0],
        'e in': [-1],
        'e out': [-1 ]
    }
    den = den*y
    alpha, index = quantize(den, grd_lvl, "unif")
    alpha = alpha.reshape(1,-1)
    gradient = - Master.uniform_query(alpha, grd_lvl, dict, X, index)/len(y)

    working_loss, working_gradient = working_vers(w, Master, y, X)
    if not np.allclose(gradient, working_gradient):
        raise ValueError(f"bad gradient, {np.hstack((gradient.reshape(-1, 1), working_gradient.reshape(-1, 1)))[0:5]}")
    if not np.allclose(loss, working_loss):
        raise ValueError(f"bad loss, {loss, working_loss}")

    record_access(dict, filename)
    gradient = gradient.reshape(-1,1)
    return loss, gradient



def record_access(dict, filename):

    df = pd.DataFrame(dict)
    df.to_csv(filename, mode='a', index=False, header=False)

def working_vers(w, Master,yTr, X):
    dictionary = {}
    y_pred = Master.query(w, X, dictionary)
    loss = np.sum(np.log(1 + np.exp(-yTr * y_pred)))
    den = (1 + np.exp(yTr * y_pred))
    alpha = yTr / den
    gradient = - Master.query(alpha.reshape(1, -1), X, dictionary).reshape(-1, 1)/len(yTr)
    return loss, gradient