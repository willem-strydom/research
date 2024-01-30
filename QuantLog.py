import numpy as np
from quantize import quantize
from query import query
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


def quantlogistic(w,xTr,yTr,num_bins, type, nodes_array):
    #y_pred = w.T @ xTr ... now with low access
    y_pred = query(w,nodes_array).reshape(-1,1)
    vals = yTr * y_pred
    #keeping same loss function as for normal log loss?
    loss = np.mean(np.log(1 + np.exp(-vals)))

    # take sigmoid of vals

    func = lambda x: 1/(1+np.exp(x))
    func = np.vectorize(func)
    vals = func(vals)

    # then quantize

    #beta = quantize(vals, num_bins, type)
    beta = np.sign(vals)
    gradient = -np.mean(yTr * xTr * beta, axis = 1).reshape(-1, 1)

    # store the values for later analysis of distribution...

    """file_path = f'values{num_bins}.csv'
    with open(file_path, "a") as f:
        np.savetxt(f, vals, delimiter=',')
"""
    return loss, gradient