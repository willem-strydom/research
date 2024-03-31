import math
import numpy as np

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
def normallogistic(w,xTr,yTr):
    y_pred = w.T @ xTr
    loss = np.sum(np.log(1 + np.exp(-yTr * y_pred)))
    num = yTr * xTr
    den = (1 + np.exp(yTr * (y_pred)))
    gradient = -np.sum((num / den), axis=1).reshape(-1, 1)

    return loss,gradient