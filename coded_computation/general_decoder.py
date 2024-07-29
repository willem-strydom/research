import numpy as np
def general_decoder(B):
    """
    Param: B: non-systematic generator matrix of code
    :return: lookup table for code... should work for any code that is of the form G = [i|B]
    """

    if B is None:
        return None
    B = B.T
    len = B.shape[1]
    all_combinations = np.array(np.meshgrid(*[[-1, 1]] * len)).T.reshape(-1, len)  # vectors in rows
    lookup_table = {}
    for v in all_combinations:
        best = len # maximum hamming distance between v and b
        best_code = 0
        best_ind = 0
        sign = 0
        for i,b in enumerate(B):
            dummy = np.where(v != b, 1, 0)
            dist = np.sum(dummy)

            if dist < best:
                best = dist
                best_code = b
                best_ind = i
                sign = 1

            dummy = np.where(-v != b, 1, 0)
            dist = np.sum(dummy)
            if dist < best:
                best = dist
                best_code = -b
                best_ind = i
                sign = -1

        correction = v - best_code
        code = np.zeros(B.shape[0])
        code[best_ind] = sign
        lookup_table[tuple(v)] = np.hstack((correction,code))
    return lookup_table

"""
G = np.array([
        [1,1,1,1,1,1,1],
        [-1,-1,-1,1,1,1,1],
        [-1,1,1,-1,-1,1,1],
        [1,-1,-1,-1,-1,1,1],
        [1,-1,1,-1,1,-1,1],
        [-1,1,-1,-1,1,-1,1],
        [-1,-1,1,1,-1,-1,1],
        [1,1,-1,1,-1,-1,1]
    ])
table = general_decoder(G.T)

"""