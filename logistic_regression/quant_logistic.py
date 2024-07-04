import numpy as np
from quantization.quantize import quantize
from logistic_regression.stable_logistic import stable_sigmoid, stable_loss
from util import record_access

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
    :return: loss, gradient as numpy column vector
    """
    y_pred = Master.uniform_query(w, w_lvl, dict, X, index)
    record_access(dict, filename)

    vals = y * y_pred
    loss = stable_loss(-vals)  # computation of loss robust to overflow
    func = np.vectorize(stable_sigmoid)
    den = func(-vals)  # computes 1/1+e^{-x}, so should passs -vals for correct computation
    alpha = den*y
    alpha, index = quantize(alpha, grd_lvl, "unif")
    alpha = alpha.reshape(1,-1)  # alpha is a row vector
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
        'e out': [-1]
    }
    gradient = - Master.uniform_query(alpha, grd_lvl, dict, X, index)/len(y)
    record_access(dict, filename)
    gradient = gradient.reshape(-1,1)  # gradient, model are column vectors always
    return loss, gradient
