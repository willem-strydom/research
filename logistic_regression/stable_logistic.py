import numpy as np

def stable_loss(x):
    # compute loss robustly to avoid overflow
    """
    :param x: values
    :return: logistic loss
    """
    max_val = np.max(x)
    epsilon = 1e-10
    result = np.sum(max_val + np.log(np.exp(-max_val) + np.exp(x - max_val) + epsilon))

    return result


def stable_sigmoid(x):
    # compute sigmoid function = 1/1+e^{-x} robustly to avoid overflow
    """
    :param x: value
    :return: sigmoid(x)
    """
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)
