import numpy as np
from logistic_regression.GrdDscntQuant import grdescentquant
from logistic_regression.normal_logistic import normallogistic
from logistic_regression.GrdDscnt import grdescentnormal
from logistic_regression.quant_logistic import quant_logistic
from logistic_regression.GrdDscntUncoded import grdescentuncoded
from logistic_regression.uncoded_logistic import  uncoded_logistic
from quantization.quantize import quantize
from coded_computation.master import master
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from pad_and_clean import pad
from pad_and_clean import clean_and_scale
import time
from gen_data import gen_data, gen_nonlinear_data
from sklearn.model_selection import train_test_split

def get_loss(w,X,y):
    #calculates 1-0 prediction error
    log_odds = X@w
    probs = 1 / (1 + np.exp(-log_odds))
    preds = np.where(probs > 0.5, 1,-1)
    test_loss = np.mean(preds != y)

    return test_loss
def plot_3d_bar(data, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the x, y coordinates and the z heights
    _x = np.arange(data.shape[1])
    _y = np.arange(data.shape[0])
    _x, _y = np.meshgrid(_x, _y)
    x, y = _x.ravel(), _y.ravel()

    # The z values represent the bar heights
    z = np.zeros_like(x)
    dz = data.ravel()

    # Plot 3D bars
    ax.bar3d(x, y, z, 1, 1, dz, shade=True)

    # Labels
    ax.set_xlabel('w_lvl')
    ax.set_ylabel('grd_lvl')
    ax.set_zlabel(z)

    plt.show()
"""
data = arff.loadarff('/Users/willem/Downloads/speeddating.arff')
df = pd.DataFrame(data[0])

hill_train_x, hill_test_x, hill_train_y, hill_test_y = clean_and_scale(df, "match")
hill_train_x, hill_train_y = pad(hill_train_x, hill_train_y, 7)
hill_test_x, hill_test_y = pad(hill_test_x, hill_test_y, 7)

print(hill_train_x.shape, hill_train_y.shape, hill_test_x.shape, hill_test_y.shape)

datasets = [(hill_train_x, hill_train_y, hill_test_x, hill_test_y),]"""

repetitions = 10
func = quant_logistic
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

stepsize = 0.1
maxiter = 10000

X,y = gen_nonlinear_data(500, 40, 1)
print(X.shape,y.shape)
X, Xt, y, yt = train_test_split(X, y, test_size=0.2)
print(X.shape, y.shape)
X, y = pad(X, y, 7)
Xt, yt = pad(Xt, yt, 7)
print(X.shape, y.shape)
Master_uncoded = master(X, None, 21)
Master = master(X, G, 3)
times_grid = np.zeros((5, 4))
loss_grid = np.zeros((5, 4))  # Initialize for each dataset
iters_grid = np.zeros((5, 4))
test_loss = np.zeros((5, 4))
test_loss_uncoded = np.zeros((5,4))
# loss from normal logistic regression
w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
start_time = time.time()
w, num_iters = grdescentnormal(normallogistic, w0, stepsize, maxiter, Master_uncoded, y, X, tolerance=1e-02)
normal_loss = get_loss(w, Xt, yt)
end_time = time.time()
print(f"loss from unquantized logistic regresison: {normal_loss} on {num_iters} iterations in {end_time - start_time} seconds")
unquantized_times = np.zeros((5, 4))
for i in range(repetitions):
    for w_lvl in range(4, 9):
        for grd_lvl in range(2, 6):
            # logic for calculations
            w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
            w0 = quantize(w0, w_lvl, "unif")

            start_time = time.time()
            w, num_iters = grdescentquant(func, w0, stepsize, maxiter, Master, w_lvl, grd_lvl, X, y, tolerance=1e-02)
            end_time = time.time()

            times_grid[w_lvl - 4, grd_lvl - 2] += end_time - start_time
            loss_grid[w_lvl - 4, grd_lvl - 2] += get_loss(w, X, y)
            iters_grid[w_lvl - 4, grd_lvl - 2] += num_iters
            test_loss[w_lvl - 4, grd_lvl - 2] += get_loss(w, Xt, yt)

            start_time = time.time()
            w, num_iters = grdescentuncoded(uncoded_logistic, w0, stepsize, maxiter, Master_uncoded, w_lvl, grd_lvl, X, y, tolerance=1e-02)
            end_time = time.time()
            unquantized_times[w_lvl - 4, grd_lvl - 2] = end_time - start_time
            test_loss_uncoded[w_lvl - 4, grd_lvl - 2] = get_loss(w,Xt,yt)

test_loss = test_loss/repetitions
loss_grid = loss_grid/repetitions
iters_grid = iters_grid/repetitions
times_grid = times_grid/repetitions


print(f" training loss: \n{loss_grid}")
print(f" avg iterations: \n{iters_grid}")
print(f" test loss: \n{test_loss}")
print(f"run time: \n{times_grid}")
