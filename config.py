"""
config file: never actually used one of these before,
but I thought it might make it easier to have global variables like
the Master instance, and other stuff here instead of having to pass them around between files...
maybe not a good way to do this though.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gen_data import gen_data

scaler = MinMaxScaler(feature_range=(-1, 1))

X, y = gen_data(840, 31, 4)  # ends up giving 210 features
X_test = X[700:,:]
y_test = y[700:]
X = scaler.fit_transform(X)[:700,:]
y = np.array(y).reshape(-1, 1)[:700]

