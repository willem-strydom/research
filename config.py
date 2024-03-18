"""
config file: never actually used one of these before,
but I thought it might make it easier to have global variables like
the Master instance, and other stuff here instead of having to pass them around between files...
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from gen_data import gen_data
from master import master

scaler = MinMaxScaler(feature_range=(-1, 1))

grd_lvl = 3
w_lvl = 3
X, y = gen_data(210, 190)  # ends up giving 210 features
X = scaler.fit_transform(X)
y = np.array(y).reshape(-1, 1)
