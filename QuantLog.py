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


def quantlogistic(w,xTr,yTr, nodes_array):
    #y_pred = w.T @ xTr ... now with low access
    y_pred = query(w,nodes_array[0])[0]
    vals = yTr.flatten() * y_pred
    loss = np.mean(np.log(1 + np.exp(-vals)))

    # then quantize yi*qi
    vals = vals*yTr
    beta = np.where(vals>0,-1,1)
    beta = beta.reshape(-1,1)
    gradient = -1 * query(beta, nodes_array[1])[0]/len(yTr)
    gradient = gradient.reshape(-1,1)
    #want to find vals.T@X with the low access scheme
    #gradient = -np.mean(yTr * xTr * beta, axis = 0).reshape(-1, 1)

    return loss, gradient