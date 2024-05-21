import numpy as np

def gen_data(n,d, noise):
    """
    :param n: num examples
    :param d: num indep features... makes int(d/10) dependent features too
    :param noise: scale param for 0 mean gaussian noise added to labels
    :return: np nd-rray
    """
    # indep features
    X = np.ones((n,1)) # first feature for bias
    for i in range(d):
        feature = np.random.normal(loc=0, scale = np.random.uniform(1,4), size=(n,1))
        X = np.hstack((X,feature))

    # generate dependent features
    M = np.random.rand(d+1,int(d/10))
    dp_features = X @ M
    X = np.hstack((X,dp_features))

    w_star = np.random.normal(0,1,X.shape[1])

    y = np.sign((X@ w_star) + np.random.normal(0, noise, X.shape[0])).reshape(-1,1)

    return X,y


def gen_nonlinear_data(n,d, noise):
    """
    :param n: num examples
    :param d: num independent features... makes int(d/10) dependent features too
    :param noise: scale param for 0 mean gaussian noise added to labels
    :return: np nd-rray
    """
    # indep features
    X = np.ones((n,1)) # first feature for bias
    for i in range(d):
        feature = np.random.normal(loc=0, scale = np.random.uniform(1,4), size=(n,1))
        X = np.hstack((X,feature))


    # generate dependent features
    M = np.random.rand(d+1,int(d/10))
    dp_features = X @ M
    X = np.hstack((X,dp_features))

    w_star = np.random.normal(0,1,X.shape[1])
    num_linear_features = int(X.shape[1]*0.8)
    y = np.sign(
        X[:,:num_linear_features] @ w_star[:num_linear_features] +
        (X[:,num_linear_features:]**2) @ w_star[num_linear_features:] +
        np.random.normal(0, noise, X.shape[0])
    ).reshape(-1,1)

    return X,y
"""test to see if the noisy label data is actually being done correctly, need to return w though
X, y, w = gen_data(600,5, 10)

preds = np.sign(X@w)
print(np.mean(y == preds))

n = 1000
d = 32
noise = 1
X,y  = gen_nonlinear_data(n,d, noise)
print(X.shape, y.shape)"""