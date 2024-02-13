import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def normallogistic(w,xTr,yTr):

    y_pred = w.T@xTr
    loss = np.mean(np.log(1 + np.exp(-yTr * y_pred)))
    num = yTr*xTr
    den = (1 + np.exp(yTr * y_pred))
    gradient = -np.mean((num / den), axis=1).reshape(-1, 1)

    return loss,gradient