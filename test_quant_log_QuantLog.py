import numpy as np
from test_quant_log_quantize import quantize

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
import csv


def quantlogistic(w,xTr,yTr,num_bins, type):

    y_pred = w.T @ xTr
    vals = yTr * y_pred
    #keeping same loss function as for normal log loss?
    loss = np.mean(np.log(1 + np.exp(-vals)))

    # take sigmoid of vals

    func = lambda x: 1/(1+np.exp(x))
    func = np.vectorize(func)
    vals = func(vals)

    # then quantize

    beta = quantize(vals, num_bins, type)
    gradient = -np.mean(yTr * xTr * beta, axis = 1).reshape(-1, 1)

    # store the values for later analysis of distribution...

    """file_path = f'values{num_bins}.csv'
    with open(file_path, "a") as f:
        np.savetxt(f, vals, delimiter=',')
"""
    return loss, gradient