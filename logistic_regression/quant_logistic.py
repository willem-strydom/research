import numpy as np
import pandas as pd

from quantization.quantize import quantize
from coded_computation.uniform_query import uniform_query
import csv

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


def quant_logistic(w, Master, w_lvl, grd_lvl, dict, X, y, filename):
    #y_pred = w.T @ xTr ... now with low access
    y_pred = uniform_query(w, Master, w_lvl, dict, X)
    vals = y * y_pred
    loss = np.mean(np.log(1 + np.exp(-vals)))
    record_access(dict, filename)

    func = lambda x: 1 / (1 + np.exp(x))
    func = np.vectorize(func)
    vals = func(vals)

    # then quantize y_i*alpha_i
    # rest dict
    dict = {
        'w-quantization': [w_lvl],
        'grd-quantization': [grd_lvl],
        'imputation': [0],
        'access': 0,
        'query type': [0],
        'time': [0],
        'stop cond': [0],
        'iters': [0]

    }
    vals = vals*y
    alpha = quantize(vals, grd_lvl, "unif").reshape(1,-1)
    gradient = - uniform_query(alpha, Master, grd_lvl, dict, X)/len(y)
    record_access(dict, filename)
    gradient = gradient.reshape(-1,1)
    return loss, gradient



def record_access(dict, filename):

    df = pd.DataFrame(dict)
    df.to_csv(filename, mode='a', index=False, header=False)

