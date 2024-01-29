import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from test_quant_log_experiment import experiment
from test_quant_log_experiment import test_loss
import random
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
from test_quant_log_gen_data import gen_data
X,y = gen_data()
X = scaler.fit_transform(X)

bias = np.ones((X.shape[1],1)).T
X = np.vstack((bias, X)).astype(float)

experiment(X,y)