import numpy as np
from quantization.quantize import quantize
from coded_computation.uniform_query import uniform_query
from config import y

'''
    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0) dx1

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''


def quant_logistic(w, Master, w_lvl, grd_lvl):
    #y_pred = w.T @ xTr ... now with low access
    y_pred = uniform_query(w, Master, w_lvl)
    vals = y * y_pred
    loss = np.mean(np.log(1 + np.exp(-vals)))

    func = lambda x: 1 / (1 + np.exp(x))
    func = np.vectorize(func)
    vals = func(vals)

    # then quantize y_i*alpha_i
    vals = vals*y
    alpha = quantize(vals, grd_lvl, "unif").reshape(1,-1)
    gradient = - uniform_query(alpha, Master, grd_lvl)/len(y)
    gradient = gradient.reshape(-1,1)
    return loss, gradient
