from logistic_regression.normal_logistic import normallogistic
from logistic_regression.GrdDscnt import grdescentnormal
import numpy as np
from coded_computation.master import master
from scipy.io import arff
from pad_and_clean import pad
from pad_and_clean import clean_and_scale
import pandas as pd

def get_loss(w,X,y):
    #calculates 1-0 prediction error
    log_odds = X@w
    probs = 1 / (1 + np.exp(-log_odds))
    preds = np.where(probs > 0.5, 1,-1)
    test_loss = np.mean(preds != y)

    return test_loss
data = arff.loadarff('/Users/willem/Downloads/speeddating.arff')
df = pd.DataFrame(data[0])

hill_train_x, hill_test_x, hill_train_y, hill_test_y = clean_and_scale(df, "match")
hill_train_x, hill_train_y = pad(hill_train_x, hill_train_y, 7)
hill_test_x, hill_test_y = pad(hill_test_x, hill_test_y, 7)

X = hill_train_x
y = hill_train_y
Xt = hill_test_x
yt = hill_test_y
G = None

Master = master(X, G, 21)
w0 = np.random.uniform(-1, 1, (X.shape[1], 1))
print(w0.shape)
maxiter = 10000
stepsize = 0.1

w, num_iters = grdescentnormal(normallogistic, w0, stepsize, maxiter, Master, y, X, tolerance=1e-02)
loss = get_loss(w, Xt, yt)
print(f"e_out from unquantized logistic regresison: {loss} on {num_iters} iterations")