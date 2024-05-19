import math
import numpy as np
from config import X

'''

    INPUT:
    xTr nxd matrix (each row is an input vector)
    yTr nx1 matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [n,d]=size(xTr);
'''
def normallogistic(w, Master,yTr):
    """y_pred = w.T @ xTr
    loss = np.sum(np.log(1 + np.exp(-yTr * y_pred)))
    num = yTr * xTr
    den = (1 + np.exp(yTr * (y_pred)))
    gradient = -np.sum((num / den), axis=1).reshape(-1, 1)"""
    dictionary = {}
    y_pred = Master.query(w, X, dictionary)
    loss = np.sum(np.log(1 + np.exp(-yTr * y_pred)))
    den = (1 + np.exp(yTr * y_pred))
    alpha = yTr / den
    gradient = - Master.query(alpha.reshape(1,-1), X, dictionary).reshape(-1,1)/len(yTr)

    """y_pred = X @ w
    num = X * yTr
    den = (1 + np.exp(yTr * (y_pred)))
    actual_gradient = -np.sum((num / den), axis=0).reshape(-1, 1)

    print(actual_gradient.shape, gradient.shape)
    if not np.allclose(actual_gradient, gradient):
        raise ValueError(f"actual, {actual_gradient[0:5]}, ours {gradient[0:5]}")"""
    return loss, gradient