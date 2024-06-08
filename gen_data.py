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

def gen_seperable_data(n,d):
    """
    :param n: num examples
    :param d: num independent features... makes int(d/10) dependent features too
    :param noise: scale param for 0 mean gaussian noise added to labels
    :return: np nd-rray
    """
    # indep features
    X = np.ones((n,1)) # first feature for bias
    for i in range(d//2):
        feature = np.random.normal(loc=0, scale = np.random.uniform(1,4), size=(n,1))
        X = np.hstack((X,feature))
    for j in range(d-d//2):
        feature = np.random.exponential(scale=np.random.uniform(1, 4), size=(n, 1))
        X = np.hstack((X, feature))


    # generate dependent features
    M = np.random.rand(d+1,int(d/10))
    dp_features = X @ M
    X = np.hstack((X,dp_features))

    w_star = np.random.normal(0,1,(X.shape[1], 1))
    y = np.sign(X@w_star).reshape(-1,1)

    return X, y, w_star

def gen_margin_seperable_data(n,d, margin):
    """
    :param n: num examples
    :param d: num independent features... makes int(d/10) dependent features too
    :param noise: scale param for 0 mean gaussian noise added to labels
    :return: np nd-rray
    """
    # indep features
    X = np.ones((n,1)) # first feature for bias
    for i in range(d//2):
        feature = np.random.normal(loc=0, scale = np.random.uniform(1,4), size=(n,1))
        X = np.hstack((X,feature))
    for j in range(d-d//2):
        feature = np.random.exponential(scale=np.random.uniform(1, 4), size=(n, 1))
        X = np.hstack((X, feature))


    # generate dependent features
    M = np.random.rand(d+1,int(d/10))
    dp_features = X @ M
    X = np.hstack((X,dp_features))

    w_star = np.random.normal(0,1,(X.shape[1], 1))
    y = np.sign(X@w_star).reshape(-1,1)
    distance_index = np.ones(X.shape[0], dtype=bool)
    i = 0
    for x in X:
        # remove points that are too close to the decision boundary by making index
        distance = np.abs(np.dot(x, w_star)) / np.linalg.norm(w_star)
        print(distance)
        if distance < margin:
            distance_index[i] = False
        i += 1
    X = X[distance_index]
    y = y[distance_index]
    return X, y, w_star
