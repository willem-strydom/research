from node import node
import numpy as np
from general_decoder import general_decoder
from query import query


class master:

    def __init__(self,data,G):

        self.nodes_array = make_nodes(data,G)
        self.col_parity = data@np.ones(data.shape[1])
        self.row_parity = np.ones(data.shape[0])@data


def make_nodes(data: np.ndarray, G: np.ndarray):
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
    n = int(data.shape[1]/width)
    # number of row nodes
    m = int(data.shape[0]/width)
    decoder = general_decoder(G[:,width:].T)

    # init nodes and partition data.
    # rows and cols must both be divisible by 7 according to the new scheme.
    # Also, I think that I need to know store the nodes in a more clever way maybe... what a mess
    nodes_array = np.empty((m, n), dtype=object)
    # make an m x n array of nodes where each  node stores some mxm array of raw data and associated parities
    for i in range(n):
        for j in range(m):
            nodes_array[j, i] = node(data[width*j:width*(1+j), width*i:width*(1+i)], decoder, G)
    # do a bunch of random queries
    return nodes_array