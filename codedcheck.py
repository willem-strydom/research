from logistic_regression.uncoded_logistic import uncoded_logistic
from logistic_regression.GrdDscntUncoded import grdescentuncoded
import numpy as np
from config import y, X
from coded_computation.master import master

def get_loss(w,X,y):
    #calculates 1-0 prediction error
    log_odds = X@w
    probs = 1 / (1 + np.exp(-log_odds))
    preds = np.where(probs > 0.5, 1,-1)
    test_loss = np.mean(preds != y)

    return test_loss

G = None

Master = master(X, G, 3)
w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
maxiter = 10000
stepsize = 0.1

w, num_iters = grdescentuncoded(uncoded_logistic, w0, stepsize, maxiter, Master, 8, 4, tolerance=1e-02)
normal_loss = get_loss(w, X, y)
print(f"loss from unquantized logistic regresison: {normal_loss} on {num_iters} iterations")