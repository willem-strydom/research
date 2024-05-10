import numpy as np
from logistic_regression.GrdDscntQuant import grdescentquant
from logistic_regression.normal_logistic import normallogistic
from logistic_regression.GrdDscnt import grdescentnormal
from logistic_regression.quant_logistic import quant_logistic
from quantization.quantize import quantize
from config import y, X, y_test, X_test
from coded_computation.master import master
import matplotlib.pyplot as plt
import pandas as pd

def get_loss(w,X,y):
    #calculates 1-0 prediction error
    log_odds = X@w
    probs = 1 / (1 + np.exp(-log_odds))
    preds = np.where(probs > 0.5, 1,-1)
    test_loss = np.mean(preds != y)

    return test_loss

I = np.eye(7)
G = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, 1, 1, 1, 1],
    [-1, 1, 1, -1, -1, 1, 1],
    [1, -1, -1, -1, -1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1],
    [-1, 1, -1, -1, 1, -1, 1],
    [-1, -1, 1, 1, -1, -1, 1],
    [1, 1, -1, 1, -1, -1, 1]
]).T
#G = np.hstack((I, B))
Master = master(X,G, 3)

func = quant_logistic

loss_grid = np.zeros((5, 4))  # Initialize the loss grid for storing losses
iters_grid = np.zeros((5, 4))
test_loss = np.zeros((5, 4))
maxiter = 10000
stepsize = 0.1
w0 = np.random.uniform(-1, 1, (X.shape[1 ], 1))
w, num_iters = grdescentnormal(normallogistic, w0, stepsize, maxiter, X.T, y.T, tolerance=1e-02)
normal_loss = get_loss(w, X, y)
print(f"loss from unquantized logistic regresison: {normal_loss} on {num_iters} iterations")

repetitions = 10
for i in range(repetitions):
    for w_lvl in range(3, 8):
        for grd_lvl in range(1, 5):
            # logic for calculations
            w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
            w0 = quantize(w0, w_lvl, "unif")

            print(w_lvl,grd_lvl)

            w, num_iters = grdescentquant(func, w0, 0.1, 10000, Master, w_lvl, grd_lvl, tolerance=1e-02)

            loss_grid[w_lvl - 3, grd_lvl - 1] += get_loss(w, X, y)
            iters_grid[w_lvl - 3, grd_lvl - 1] += num_iters
            test_loss[w_lvl - 3, grd_lvl - 1] += get_loss(w, X_test, y_test)

iters_grid = iters_grid/repetitions
loss_grid = loss_grid/repetitions
test_loss = test_loss/repetitions
plt.figure(figsize=(8, 8))  # Set the figure size as needed
plt.imshow(loss_grid, cmap='viridis')
plt.xlabel("gradient lvl(1-4)")
plt.ylabel("w lvl (3-7): 7 on bottom")
plt.colorbar()  # Optionally add a colorbar to indicate the scale
plt.show()

plt.figure(figsize=(8, 8))  # Set the figure size as needed
plt.imshow(test_loss, cmap='viridis')
plt.xlabel("gradient lvl(1-4)")
plt.ylabel("w lvl (3-7): 7 on bottom")
plt.colorbar()  # Optionally add a colorbar to indicate the scale
plt.show()
print(normal_loss)
print(loss_grid)
print(iters_grid)

#visual for access , iterations