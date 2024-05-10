import numpy as np
from quantization.quantize import quantize
import pandas as pd
import time
def grdescentquant(func, w, stepsize, maxiter, Master, w_lvl, grd_lvl, tolerance=1e-02):
    """
    :param func: quantlog function
    :param w0: usually uniformly random in {-1,1}^d
    :param stepsize: initial learning rate, rate is modular in implementation
    :param maxiter: maximum iterations until hardstop
    :param xTr: train data.... vestigial at this point
    :param yTr: train labels
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
        dict = {
                'w-quantization': [w_lvl],
                'grd-quantization': [grd_lvl],
                'imputation': [0],
                'access': 0,
                'query type': [0],
                'time': [0],
                'stop cond': [0],
                'iters': [0]

            } # for data collection I think
        loss, gradient = func(w, Master, w_lvl, grd_lvl, dict)
        if loss > prior_loss:

            w = w + stepsize * prior_gradient
            stepsize = (stepsize / 1.01) * 0.5
            w = w - stepsize * prior_gradient
        else:
            if num_iter < 10:
                stepsize = stepsize * 1.1
                w = w - stepsize * gradient
            else:
                stepsize = stepsize * 1.01
                w = w - stepsize * gradient
        if stepsize < eps:
            stopcond = 1
            break
        if np.linalg.norm(gradient) < tolerance:
            stopcond = 2
            break
        w = quantize(w, w_lvl, "unif").reshape(-1, 1)
        if np.array_equal(gradient,prior_gradient):
            stopcond = 3
            num_iter +=1
            break
        prior_loss = loss
        prior_gradient = gradient.copy()
        num_iter += 1
    end_time = time.time()
    t = end_time - start_time
    dict = {
        'w-quantization': [w_lvl],
        'grd-quantization': [grd_lvl],
        'imputation': [0],
        'access': [0],
        'query type': [0],
        'time': [t],
        'stop cond': [stopcond],
        'iters': [num_iter]

    }
    record_access(dict)

    return w, num_iter

def record_access(dict):
    filename = 'access_measurements.csv'

    df = pd.DataFrame(dict)
    df.to_csv(filename, mode='a', index=False, header=False)

