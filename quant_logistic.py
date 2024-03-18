import numpy as np
from quantize import quantize
from general_query import general_query
from uniform_query import uniform_query
from config import y, grd_lvl
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


def quant_logistic(w, Master):
    #y_pred = w.T @ xTr ... now with low access
    y_pred = uniform_query(w, Master)
    vals = y.reshape(y.shape) * y_pred
    loss = np.mean(np.log(1 + np.exp(-vals)))

    # then quantize y_i*alpha_i
    vals = vals*y
    alpha = quantize(vals, grd_lvl, "unif").reshape(1,-1)
    gradient = -1 * uniform_query(alpha, Master)/len(y) # change to do the 1,0 scheme
    gradient = gradient.reshape(-1,1)
    #want to find vals.T@X with the low access scheme
    #gradient = -np.mean(yTr * xTr * alpha, axis = 0).reshape(-1, 1)

    return loss, gradient
