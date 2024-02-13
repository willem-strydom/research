import numpy as np
from quantize import quantize
from general_query import general_query
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


def quant_logistic(w, xTr, yTr, master):
    #y_pred = w.T @ xTr ... now with low access
    y_pred = general_query(w,master)
    vals = yTr.flatten() * y_pred
    loss = np.mean(np.log(1 + np.exp(-vals)))

    # then quantize yi*qi
    vals = vals*yTr # also need to redo this part of the query though... so cool
    alpha = np.where(vals>0, 0,1)
    alpha = alpha.reshape(1,-1)
    gradient = -1 * general_query(alpha, master)/len(yTr) # change to do the 1,0 scheme
    gradient = gradient.reshape(-1,1)
    #want to find vals.T@X with the low access scheme
    #gradient = -np.mean(yTr * xTr * alpha, axis = 0).reshape(-1, 1)

    return loss, gradient