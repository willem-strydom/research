from logistic_regression.normal_logistic import normallogistic
from logistic_regression.GrdDscnt import grdescentnormal
import numpy as np
from config import hill_train_x, hill_train_y, hill_test_x, hill_test_y
from coded_computation.master import master

def get_loss(w,X,y):
    #calculates 1-0 prediction error
    log_odds = X@w
    probs = 1 / (1 + np.exp(-log_odds))
    preds = np.where(probs > 0.5, 1,-1)
    test_loss = np.mean(preds != y)

    return test_loss
X = hill_train_x
y = hill_train_y
Xt = hill_test_x
yt = hill_test_y

print(X.shape, y.shape)
G = None

Master = master(X, G, 3)
w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
print(w0.shape)
maxiter = 10000
stepsize = 0.1

w, num_iters = grdescentnormal(normallogistic, w0, stepsize, maxiter, Master, y, tolerance=1e-02)
normal_loss = get_loss(w, Xt, yt)
print(f"loss from unquantized logistic regresison: {normal_loss} on {num_iters} iterations")