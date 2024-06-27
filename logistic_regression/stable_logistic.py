import numpy as np

def stable_loss(x):
    # compute loss robustly to avoid overflow
    """
    :param x: values
    :return: logistic loss
    """
    max_val = np.max(x)
    return np.sum(max_val + np.log(np.exp(-max_val) + np.exp(x - max_val)))


def stable_sigmoid(x):
    # compute sigmoid function robustly to avoid overflow
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

# check to make sure all same

for _ in range(100000):
    values = np.random.normal(loc=0.0, scale=100.0, size=100)
    stable_l = stable_loss(values)
    normal_l = np.sum(np.log(1 + np.exp(values)))

    func = np.vectorize(stable_sigmoid)
    stable_sig = func(values)
    normal_sig = 1/(1+np.exp(-values))
    if not np.allclose(stable_sig, normal_sig):
        print('oopsies')
        print(np.allclose(stable_sig, normal_sig))
        print(stable_sig.shape, normal_sig.shape)
    if np.any(1/stable_sig == np.nan):
        print("divide by 0 error")