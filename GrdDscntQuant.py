import numpy as np
from quantize import quantize
from master import master
def grdescentquant(func, w0, stepsize, maxiter, xTr, yTr,nodes_array, tolerance=1e-02):
    # INPUT:
    # func function to minimize
    # w_trained = initial weight vector
    # stepsize = initial gradient descent stepsize
    # tolerance = if norm(gradient)<tolerance, it quits
    #
    # OUTPUTS:
    #
    # w = final weight vector
    eps = 2.2204e-14  # minimum step size for gradient descent

    # YOUR CODE HERE
    num_iter = 0
    w = w0
    gradient = 0
    prior_gradient = np.zeros(w.shape)
    # why init to 0? should be large number or else bad no?
    prior_loss = 1e06
    # Increase the stepsize by a factor of 1.01 each iteration where the loss goes down,
    # and decrease it by a factor 0.5 if the loss went up. ...
    # also undo the last update in that case to make sure
    # the loss decreases every iteration

    while num_iter < maxiter:
        loss, gradient = func(w, xTr, yTr, nodes_array)
        print(f"gradient: {gradient.shape}")
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
            break
        if np.linalg.norm(gradient) < tolerance:
            break
        #w = quantize(w, level_w, type_w).reshape(-1, 1)
        w = np.sign(w)
        print(f"w: {w.shape}")
        if np.array_equal(gradient,prior_gradient):
            num_iter +=1
            break
        prior_loss = loss
        prior_gradient = gradient.copy()
        num_iter += 1

    return w, num_iter