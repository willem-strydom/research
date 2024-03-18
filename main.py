import numpy as np
from GrdDscntQuant import grdescentquant
from quant_logistic import quant_logistic
from quantize import quantize
from config import y, X
from master import master

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

w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
w0 = quantize(w0,2,"unif")
# make the query table and store it somewhere

w, num_iters = grdescentquant(func, w0, 0.1, 10000, Master, tolerance=1e-02)
print(num_iters)
print(w.shape)
preds = np.sign(np.dot(X,w))
err = np.mean(y != [preds])
print(err)
