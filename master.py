from node import node
import numpy as np
from query import query
def master(m: int, data: np.ndarray, decoder: dict, G: np.ndarray):
    """

    :param m: number of instances of class node
    :param G: encoding matrix, shape data.shape[0],m I think. or maybe not same for each
    :param data: assume m| num cols of data, give each to a node
    :return: list of nodes
    """

    # init nodes and partition data
    nodes_array = []
    width = int(data.shape[1]/m) # number of cols of data stored at each node
    for i in range(m):
        # use most intuitive partition scheme
        nodes_array.append(node(data[:, width*i:width*(1+i)], decoder, G))
    # do a bunch of random queries
    accesses = []
    for i in range(500):
        w = np.random.choice([-1, 1], data.shape[1])
        ans, access = query(w, nodes_array, data.shape[0]) # needs to know how many examples
        accesses.append(access)
    print(np.mean(accesses))
    return nodes_array #not sure if this is like a one and done deal