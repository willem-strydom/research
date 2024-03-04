import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from quant_log_experiment import experiment
from quant_log_experiment import test_loss
import random
from GrdDscntQuant import grdescentquant
from quant_logistic import quant_logistic
from gen_data import gen_data
from general_decoder import general_decoder
from master import master
from quantize import quantize
from config import Master, y

func = quant_logistic

w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
w0 = quantize(w0,2,"unif")
# make the query table and store it somewhere

w, num_iters = grdescentquant(func, w0, 0.1, 10000, tolerance=1e-02)
print(num_iters)
print(w.shape)
# preds = np.sign(np.dot(X,w)) ... change this to a query
err = np.mean(y != [preds])
print(err)
