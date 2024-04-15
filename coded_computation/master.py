from node import node
import numpy as np
from general_decoder import general_decoder
from query import query


class master:

    def __init__(self, data, G, coded_cols_per_node):

        self.nodes_array = self.make_nodes(data, G, coded_cols_per_node)
        # now I need to add the column parities to the nodes and then change the uniform query function to handle that
        self.col_parity = (data@np.ones(data.shape[1])).reshape(-1,1)
        self.row_parity = (np.ones(data.shape[0])@data).reshape(1,-1)

    def query(self, w, X):
        """
        :param w: query on RAW data in {-1,1}^n
        :param nodes_array: the storage nodes from master in an mxn array
        :return: np.dot(w,data) but like with low access
        """
        nodes_array = self.nodes_array
        rows, cols = nodes_array.shape

        width = nodes_array[0, 0].G.shape[
            0]  # size of raw data at each node is needed to partition w. assume systematic B
        # partition w and do a query
        # access = np.zeros(m) im not going to bother computing access, its annoying and we already saw that its about 2 per node
        # ans_array, access[0] = nodes_array[0].query(w[width*0:width*(0+1)])

        # linear comb of rows
        if w.shape[0] == 1:
            ans = np.zeros((1, cols * width))
            for m in range(rows):
                response = []
                for n in range(cols):
                    response.append(nodes_array[m, n].query(w[:, width * m:width * (1 + m)]))
                    # response.append(nodes_array[m,n].query(w[width*m:width*(1+m)].reshape(1,-1)))
                response = np.hstack(response).reshape(1, -1)
                ans += response

            expected = w @ X
            if not np.allclose(ans, expected):
                raise ValueError(
                    f"wrong +- query: expected {expected.shape}, got {ans.shape}. with error: {np.abs(expected - ans)[0:5]}")
            return ans

        # linear comb of cols
        if w.shape[1] == 1:
            ans = np.zeros((rows * width, 1))
            for n in range(cols):
                response = []
                for m in range(rows):
                    response.append(nodes_array[m, n].query(w[width * n:width * (1 + n), :]))
                    # response.append(nodes_array[m,n].query(w[width*n:width*(1+n)].reshape(-1,1)))
                response = np.hstack(response).reshape(-1, 1)
                ans += response

            expected = X @ w
            if not np.allclose(ans, expected):
                raise ValueError(
                    f"wrong +- query: expected {expected.shape}, got {ans.shape}. with error: {np.abs(expected - ans)[0:5]}")
            return ans


    def make_nodes(self, data, G, coded_cols_per_node):
        """
        codes data into a square matrix of "nodes", where each node stores a mXm cut of the data, and the associated
        row and column parities defined by G. This protocol is for any code defined by g^mxn I do believe... reeee

        :param G: non - systematic encoding matrix, shape data.shape[0],m
        :param data: assume m| num cols of data, give each to a node
        :return: array of nodes
        """
        # ammount of data at each node
        width = G.shape[0] * coded_cols_per_node
        # number of column nodes
        n = int(data.shape[1]/width)
        decoder = general_decoder(G.T)

        # init nodes and partition data.
        # rows and cols must both be divisible by 7 according to the new scheme.
        # Also, I think that I need to know store the nodes in a more clever way maybe... what a mess
        nodes_list = np.empty(n, dtype=object)
        # make an n length list of nodes where each  node stores some slice of the columns of the data
        for i in range(n):
            nodes_list[i] = node(data[:, i*width, width*(i+1)], decoder, G)
        # do a bunch of random queries
        return nodes_list
