import numpy as np
from binning import binning

def quantize_w(w,num_bins):
    """

    :param w: w
    :param num_bins: log2 number of bins
    :return: quantized w
    """
    bins = binning(w,num_bins)
    alpha = np.digitize(w,bins).flatten()
    N = len(bins)
    beta = np.zeros(alpha.shape)
    i = 0
    for a in alpha[:-1]:
        if a == 0:
            beta[i] = bins[a]
        if a == N:
            beta[i] = bins[a]
        else:
            beta[i] = (bins[a] +bins[a+1]) /2


