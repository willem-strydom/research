import numpy as np
def get_loss(w,X,y):
    #calculates 1-0 prediction error
    log_odds = X@w
    probs = 1 / (1 + np.exp(-log_odds))
    preds = np.where(probs > 0.5, 1,-1)

    return np.mean(preds != y)
