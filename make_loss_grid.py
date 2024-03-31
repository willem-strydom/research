import numpy as np
import importlib
from GrdDscntQuant import grdescentquant
from normal_logistic import normallogistic
from GrdDscnt import grdescentnormal
from quant_logistic import quant_logistic
from quantize import quantize
from config import y, X
from master import master
import matplotlib.pyplot as plt

def get_loss(w,X,y):
    #calculates 1-0 prediction error
    log_odds = X@w
    probs = 1 / (1 + np.exp(-log_odds))
    preds = (probs > 0.5).astype(int)
    preds = np.where(preds == 0, -1, preds)
    test_loss = np.sum(preds != y) / len(y)

    return test_loss

# The path to your config.py file
config_path = '/Users/willem/Desktop/research/low_access-merge_tql/quant_lvls.py'
I = np.eye(7)
B = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, 1, 1, 1, 1],
    [-1, 1, 1, -1, -1, 1, 1],
    [1, -1, -1, -1, -1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1],
    [-1, 1, -1, -1, 1, -1, 1],
    [-1, -1, 1, 1, -1, -1, 1],
    [1, 1, -1, 1, -1, -1, 1]
]).T
G = np.hstack((I, B))
Master = master(X,G)

func = quant_logistic

loss_grid = np.zeros((7,4))

loss_grid = np.zeros((7, 4))  # Initialize the loss grid for storing losses

maxiter = 10000
stepsize = 0.1
w0 = np.random.uniform(-1, 1, (X.shape[1 ], 1))
w, num_iters = grdescentnormal(normallogistic, w0, stepsize, maxiter, X.T, y.T, tolerance=1e-02)
normal_loss = get_loss(w, X, y)
print(f"loss from unquantized logistic regresison: {normal_loss}")

for w_lvl in range(1, 8):
    for grd_lvl in range(1, 5):
        """# Import the config module
        config_module = importlib.import_module("quant_lvls")

        # Update w_lvl and grd_lvl in the config module
        config_module.w_lvl = w_lvl
        config_module.grd_lvl = grd_lvl

        # Convert the updated configuration into a string
        new_content = f"w_lvl = {w_lvl}\ngrd_lvl = {grd_lvl}\n"

        # Write the new content to the config.py file
        with open(config_path, 'w') as config_file:
            config_file.write(new_content)

        # Reload the module to reflect the updates from the file
        importlib.reload(config_module)"""

        # Your existing logic for calculations
        w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
        w0 = quantize(w0, w_lvl, "unif")

        print(w_lvl,grd_lvl)

        w, num_iters = grdescentquant(func, w0, 0.1, 10000, Master, w_lvl, grd_lvl, tolerance=1e-02)
        loss = get_loss(w, X, y)
        loss_grid[w_lvl - 1, grd_lvl - 1] = loss

plt.figure(figsize=(8, 8))  # Set the figure size as needed
plt.imshow(loss_grid, cmap='viridis', interpolation='nearest')  # Choose a colormap that fits your needs, 'viridis' is just an example
plt.colorbar()  # Optionally add a colorbar to indicate the scale
plt.show()
print(normal_loss)
print(loss_grid)

