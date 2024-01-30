import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from quant_log_experiment import experiment
from quant_log_experiment import test_loss
import random
from GrdDscntQuant import grdescentquant
from QuantLog import quantlogistic
scaler = MinMaxScaler(feature_range=(-1, 1))
"""
#loading and sorting the data
diabetes_data = pd.read_csv("diabetes.csv").to_numpy()
diabetes_x = diabetes_data[:,:-1]
diabetes_y = diabetes_data[:,-1]
diabetes_y = np.where(diabetes_y == 0,-1, diabetes_y)
# avoid overflow error

diabetes_x = scaler.fit_transform(diabetes_x)
bias = np.ones((diabetes_x.shape[0],1))
diabetes_x = np.hstack((bias,diabetes_x))
"""
from gen_data import gen_data
X,y = gen_data(200,12)
X = scaler.fit_transform(X)
X = X.T
y = y.T
func = quantlogistic
w0 = np.random.uniform(-1, 1, (X.shape[0], 1))

w,num_iters = grdescentquant(func, w0, 0.1, 10000, X, y, level_w, level_q, type_w, type_q, tolerance=1e-02)