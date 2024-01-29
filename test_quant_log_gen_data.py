import numpy as np

def gen_data():
    N = 5000
    a1 = np.random.normal(loc=0, scale=0.9 ** 0.5, size=N)
    a2 = np.random.normal(loc=0, scale=0.8 ** 0.5, size=N)
    a3 = np.random.normal(loc=0, scale=0.7 ** 0.5, size=N)
    a4 = np.random.normal(loc=0, scale=0.6 ** 0.5, size=N)
    a5 = np.random.normal(loc=0, scale=0.5 ** 0.5, size=N)
    a6 = np.random.normal(loc=0, scale=0.4 ** 0.5, size=N)
    a7 = np.random.normal(loc=0, scale=0.3 ** 0.5, size=N)
    a8 = np.random.normal(loc=0, scale=0.2 ** 0.5, size=N)
    a9 = np.random.normal(loc=0, scale=0.15 ** 0.5, size=N)
    a10 = np.random.normal(loc=0, scale=0.1 ** 0.5, size=N)
    X = np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a1 * a2, a1 * a3, a2 * a3, a1 * a4, a2 * a4])

    d = X.shape[0]

    y = np.sign(np.random.uniform(-0.1, 0.1) + np.matmul(np.random.uniform(-0.5, 0.5, d), X))

    return X,y