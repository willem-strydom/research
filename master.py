from node import node
import numpy as np
from general_decoder import general_decoder
from query import query
def master(data: np.ndarray, G: np.ndarray):
    """
    codes data into a square matrix of "nodes", where each node stores a mXm cut of the data, and the associated
    row and column parities defined by G. This protocol is for any code defined by g^mxn I do believe... reeee

    :param G: encoding matrix, shape data.shape[0],m I think. or maybe not same for each
    :param data: assume m| num cols of data, give each to a node
    :return: array of nodes
    """
    # ammount of data at each node
    width = G.shape[0]
    # number of columns nodes
    m = (data.shape[1]/width).astype(int)
    # number of row nodes
    n = (data.shape[0]/width).astype(int)
    decoder = general_decoder(G[:,m:])

    # init nodes and partition data.
    # rows and cols must both be divisible by 7 according to the new scheme.
    # Also, I think that I need to know store the nodes in a more clever way maybe... what a mess
    nodes_array = np.zeros((m, n))
    # make an m x n array of nodes where each  node stores some mxm array of raw data and associated parities
    for i in range(n):
        for j in range(m):
            nodes_array[m, n] = node(data[width*j:width*(1+j), width*i:width*(1+i)], decoder, G)
    # do a bunch of random queries
    return nodes_array