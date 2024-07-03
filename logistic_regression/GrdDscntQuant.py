import numpy as np
from quantization.quantize import quantize
import pandas as pd
import time
from get_loss import get_loss
def grdescentquant(func, w, stepsize, maxiter, Master, w_lvl, grd_lvl, X, y, filename, tolerance, Xt, yt):

    """
    :param func: quantlog function
    :param w0: usually uniformly random in {-1,1}^d
    :param stepsize: initial learning rate, rate is modular in implementation
    :param maxiter: maximum iterations until hardstop
    :param X: train data.... vestigial at this point
    :param y: train labels
    :param master: train data stored in coded distributed system
    :param tolerance: The smallest gradient norm acceptable
    :return: w, num_iter
    """
    eps = 2.2204e-14  # minimum step size for gradient descent
    w = w.reshape(-1,1)
    num_iter = 0
    gradient = 0
    prior_gradient = np.zeros(w.shape)
    prior_loss = 10e06
    # Increase the stepsize by a factor of 1.01 each iteration where the loss goes down,
    # and decrease it by a factor 0.5 if the loss went up. ...
    # also undo the last update in that case to make sure
    # the loss decreases every iteration
    stopcond = 0
    start_time = time.time()
    while num_iter < maxiter:
        w, index = quantize(w, w_lvl, "unif")
        dict = {
            'w-quantization': [w_lvl],
            'grd-quantization': [grd_lvl],
            'imputation': [0],
            'access': 0,
            'query type': [0],
            'time': [0],
            'stop cond': [0],
            'iters': [0],
            'e in': [-1],
            'e out': [-1]
        }
        loss, gradient = func(w, Master, w_lvl, grd_lvl, dict, X, y, filename, index)
        if loss > prior_loss:

            w = w + stepsize * prior_gradient
            stepsize = (stepsize/ 1.1) * 0.8
            w = w - stepsize * prior_gradient
        else:
            if num_iter < 10:
                stepsize = stepsize * 1.1
                w = w - stepsize * gradient
            else:
                stepsize = stepsize * 1.1
                w = w - stepsize * gradient
        if stepsize < eps:
            stopcond = 1
            break
        if np.linalg.norm(gradient) < tolerance:
            stopcond = 2
            break
        prior_loss = loss
        prior_gradient = gradient.copy()
        num_iter += 1
    end_time = time.time()
    t = end_time - start_time
    e_in = get_loss(w,X,y)
    e_out = get_loss(w, Xt,yt)
    dict = {
        'w-quantization': [w_lvl],
        'grd-quantization': [grd_lvl],
        'imputation': [0],
        'access': [0],
        'query type': [0],
        'time': [t],
        'stop cond': [stopcond],
        'iters': [num_iter],
        'e in': [e_in],
        'e out': [e_out]
    }
    record_access(dict, filename)

    return w, num_iter

def record_access(dict, filename):

    df = pd.DataFrame(dict)
    df.to_csv(filename, mode='a', index=False, header=False)

