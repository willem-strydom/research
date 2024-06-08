import numpy as np
from quantization.quantize import quantize

def uncoded_logistic(w, Master, grd_lvl, X, y):
    """
    :param w: weights vector with bias
    :param Master: instance of master class
    :param grd_lvl: quantization level of the weights vector, needed for quantization func
    :param X: uncoded data
    :param y: labels
    :return: gradient but quantized
    """
    dictionary = {}
    y_pred = Master.query(w, X, dictionary)

    vals = y * y_pred
    loss = np.mean(np.log(1 + np.exp(-vals)))

    func = lambda x: 1 / (1 + np.exp(x))
    func = np.vectorize(func)
    vals = func(vals)

    # then quantize y_i*alpha_i
    vals = vals*y
    alpha, index = quantize(vals, grd_lvl, "unif")
    alpha = alpha.reshape(1,-1)
    gradient = - Master.query(alpha, X, dictionary)/len(y)
    gradient = gradient.reshape(-1,1)
    return loss, gradient

