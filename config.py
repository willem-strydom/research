"""
config file: never actually used one of these before,
but I thought it might make it easier to have global variables like
the Master instance, and other stuff here instead of having to pass them around between files...
maybe not a good way to do this though.
"""

from sklearn.preprocessing import MinMaxScaler
from gen_data import gen_data
import numpy as np
import pandas as pd
from scipy.io import arff
from pad_and_clean import pad
from pad_and_clean import clean_and_scale


scaler = MinMaxScaler(feature_range=(-1, 1))

X, y = gen_data(840, 31, 4)  # ends up giving 210 features
X_test = X[700:,:]
y_test = y[700:]
X = scaler.fit_transform(X)[:700,:]
y = np.array(y).reshape(-1, 1)[:700]



data = arff.loadarff('/Users/willem/Downloads/php3isjYz.arff')
df = pd.DataFrame(data[0])
hill_train_x, hill_test_x, hill_train_y, hill_test_y = clean_and_scale(df)

hill_train_x, hill_train_y = pad(hill_train_x, hill_train_y, 7)

hill_test_x, hill_test_y = pad(hill_test_x, hill_test_y, 7)

