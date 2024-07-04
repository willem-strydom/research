import numpy as np



def grdescentnormal(func, w0, stepsize, maxiter, Master, yTr, X,tolerance=1e-02):
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

    num_iter = 1
    w = w0
    prior_gradient = np.zeros(w.shape)
    prior_loss = np.inf
    # Increase the stepsize by a factor of 1.01 each iteration where the loss goes down,
    # and decrease it by a factor 0.5 if the loss went up. ...
    # also undo the last update in that case to make sure
    # the loss decreases every iteration
    while num_iter < maxiter:
        loss, gradient = func(w, Master, yTr, X)
        if loss > prior_loss:

            w = w + stepsize * prior_gradient
            stepsize = (stepsize / 1.1) * 0.8
            w = w - stepsize * prior_gradient
        else:
            if num_iter < 10:
                stepsize = stepsize * 1.1
                w = w - stepsize * gradient
            else:
                stepsize = stepsize * 1.1
                w = w - stepsize * gradient
        if stepsize < eps:
            break
        if np.linalg.norm(gradient) < tolerance:
            break
        prior_loss = loss
        prior_gradient = gradient
        num_iter += 1

    return w, num_iter
