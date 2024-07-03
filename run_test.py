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
from gen_data import gen_data, gen_nonlinear_data, gen_margin_seperable_data
from sklearn.model_selection import train_test_split
from get_loss import get_loss
def run(repetitions, X,y, filename):
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

    #X,y = gen_nonlinear_data(500, 40, 1)

    X, Xt, y, yt = train_test_split(X, y, test_size=0.2)
    X, y = pad(X, y, 7)
    Xt, yt = pad(Xt, yt, 7)

    Master_uncoded = master(X, None, 21)
    Master = master(X, G, 3)

    # loss from normal logistic regression
    for i in range(repetitions):
        w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
        print(i)
        w, iters = grdescentnormal(normallogistic, w0, stepsize, maxiter, Master_uncoded, y, X, tolerance=1e-02)
        eout = get_loss(w, Xt, yt)
        ein = get_loss(w, X, y)
        print(iters, eout, ein)
        for w_lvl in range(1, 5):
            for grd_lvl in range(1, 4):
                w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
                w, num_iters = grdescentquant(func, w0, stepsize, maxiter, Master, w_lvl, grd_lvl, X, y, filename, 1e-02, Xt,yt,ein,eout,iters)

    return 0

