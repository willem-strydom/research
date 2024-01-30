import numpy as np

def gen_data(n,d):
    """
    :param n: num examples
    :param d: num indep features... makes int(d/10) dependent features too
    :return: np nd-rray
    """
    # indep features
    X = np.ones((n,1)) # first feature for bias
    for i in range(d):
        feature = np.random.normal(loc=0, scale=0.9 ** (i/10), size=(n,1))
        X = np.hstack((X,feature))

    # generate dependent features
    M = np.random.rand(d+1,int(d/10))
    dp_features = X @ M
    X = np.hstack((X,dp_features))

    y = np.sign(X@ np.random.normal(0,1,X.shape[1]))

    return X,y
