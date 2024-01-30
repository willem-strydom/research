import numpy as np
from master import master
from general_decoder import general_decoder
def main():
    data = np.genfromtxt("framingham_cleaned_file.csv", dtype=float, comments='#', delimiter=",", skip_header=1)
    a = np.eye(3)
    b = np.ones((3,1))
    G = np.hstack((a,b))
    from decoder import decoder
    decoder = decoder(3)
    data = np.hstack((data,np.zeros((data.shape[0],2)))) # need m|data.shape[1] = num cols
    m = 6
    nodes_array = master(m, data, decoder, G)
    return 0

#for real this time
def hamming():
    from general_decoder import general_decoder
    data = np.genfromtxt("framingham_cleaned_file.csv", dtype=float, comments='#', delimiter=",", skip_header=1)
    I = np.eye(7)
    """I = np.where(I == 1, -1, I)
    I = np.where(I == 0, 1, I)""" # pretty sure this makes no sense but should clarify
    B = np.array([
        [1,1,1,1,1,1,1],
        [-1,-1,-1,1,1,1,1],
        [-1,1,1,-1,-1,1,1],
        [1,-1,-1,-1,-1,1,1],
        [1,-1,1,-1,1,-1,1],
        [-1,1,-1,-1,1,-1,1],
        [-1,-1,1,1,-1,-1,1],
        [1,1,-1,1,-1,-1,1]
    ]).T
    G = np.hstack((I,B))

    data = np.hstack((data, np.zeros((data.shape[0], 5))))  # need m*g.sys.shape = data.shape[1] = num cols
    m = 3

    decoder = general_decoder(B.T)
    nodes_array = master(m, data, decoder, G)

    return 0


def identity():
    """
    test for trivial encoding scheme w/ same params as for hamming code i think
    """
    data = np.genfromtxt("framingham_cleaned_file.csv", dtype=float, comments='#', delimiter=",", skip_header=1)
    G = np.eye(7)
    decoder = {}
    n = 7
    all_combinations = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)  # vectors in rows
    for v in all_combinations:
        decoder[tuple(v)] = v
    data = np.hstack((data, np.zeros((data.shape[0], 5))))
    m = 3
    nodes_array = master(m, data, decoder, G)
    return 0