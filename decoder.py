import numpy as np
def decoder(n):
    """
    make lookup table for single parity code
    :param n: dimension of code/lenght of querry part
    :return: lookup table dict "decoder"
    """
    decoder = {}
    all_combinations = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n) # vectors in rows
    for v in all_combinations:
        key = np.zeros(n+1)
        if np.sum(v) >=0:
            key[-1] = 1
            key[:-1] = np.where(v==-1, -2,0)
            decoder[tuple(v)] = key.astype(int)
        elif np.sum(v) < 0:
            key[-1] = -1
            key[:-1] = np.where(v==1, 2,0)
            decoder[tuple(v)] = key.astype(int)

    return decoder







