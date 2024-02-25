import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from quant_log_experiment import experiment
from quant_log_experiment import test_loss
import random
from GrdDscntQuant import grdescentquant
from quant_logistic import quant_logistic
scaler = MinMaxScaler(feature_range=(-1, 1))
from gen_data import gen_data
from general_decoder import general_decoder
from master import master





X,y = gen_data(210,190) #ends up giving 21 features
X = scaler.fit_transform(X)
func = quant_logistic
w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
y = np.array(y).reshape(-1,1)
I = np.eye(7)
B = np.array([
        [1,1,1,1,1,1,1],
        [-1,-1,-1,1,1,1,1],
        [-1,1,1,-1,-1,1,1],
        [1,-1,-1,-1,-1,1,1],
        [1,-1,1,-1,1,-1,1],
        [-1,1,-1,-1,1,-1,1],
        [-1,-1,1,1,-1,-1,1],
        [1,1,-1,1,-1,-1,1]
    ]).T
G = np.hstack((I,B))

Master = master(X,G)
w0 = np.sign(w0)
w,num_iters = grdescentquant(func, w0, 0.1, 10000, X, y, Master, tolerance=1e-02)
print(num_iters)
print(w.shape)
preds = np.sign(np.dot(X,w))
err = np.mean(y != [preds])
print(err)