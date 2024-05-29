from coded_computation.node import node
import numpy as np
from coded_computation.general_decoder import general_decoder
import pandas as pd
from coded_computation.generate_binary_matrix import generate_binary_matrix
from coded_computation.impute import is_approx_arithmetic_sequence
# from coded_computation.impute import impute


class master:

    def __init__(self, data, G, coded_slices_per_node):
        """
        :param data: np n by d array. n and d should be divisible by g.shape[0]
        :param G: non-systematic part of the generator matrix for given code
        :param coded_slices_per_node: number of coded columns stored at each node
        """
        if G is not None:
            self.width = G.shape[0] * coded_slices_per_node
        else:
            self.width = coded_slices_per_node
        self.nodes_list = self.make_nodes(data, G)
        # now I need to add the column parities to the nodes and then change the uniform query function to handle that
        self.col_parity = (data@np.ones(data.shape[1])).reshape(-1,1)
        self.row_parity = (np.ones(data.shape[0])@data).reshape(1,-1)

    def query(self, w, X, dict):
        """
        :param w: query on RAW data in {-1,1}^n, or any w for uncoded system
        :param nodes_array: the storage nodes from master in an mxn array
        :return: np.dot(w,data) but like with low access
        """
        nodes_list = self.nodes_list
        num_nodes = len(nodes_list)

        # partition w and do a query
        # access = np.zeros(m) im not going to bother computing access, its annoying and we already saw that its about 2 per node
        # ans_array, access[0] = nodes_array[0].query(w[width*0:width*(0+1)])

        # linear comb of rows
        if w.shape[0] == 1:
            ans = [] # list to append responses to
            for node in self.nodes_list:
                ans.extend(node.query(w, dict).flatten())

            ans = np.array(ans).reshape(1,-1)

            expected = w @ X
            if not np.allclose(ans, expected):
                raise ValueError(
                    f"wrong +- query: expected {expected.shape}, got {ans.shape}. with error: {np.abs(expected - ans)[0:5]}")
            return ans

        # linear comb of cols
        if w.shape[1] == 1:
            width = self.width  # the amount of data at each node in general... may differ for final node
            ans = np.zeros((self.nodes_list[0].data.shape[0], 1))
            for i, node in enumerate(self.nodes_list[:-1]):
                ans += node.query(w[i*width: width*(i+1),:], dict)
            ans += self.nodes_list[-1].query(w[width*(len(self.nodes_list)-1):,:], dict)

            expected = X @ w
            if not np.allclose(ans, expected):
                raise ValueError(
                    f"wrong +- query: expected {expected.shape}, got {ans.shape}. with error: {np.abs(expected - ans)[0:5]}")
            return ans


    def make_nodes(self, data, G):
        """
        codes data into a square matrix of "nodes", where each node stores a mXm cut of the data, and the associated
        row and column parities defined by G. This protocol is for any code defined by g^mxn I do believe... reeee

        :param G: non - systematic encoding matrix, shape data.shape[0],m
        :param data: assume m| num cols of data, give each to a node
        :return: array of nodes
        """
        # ammount of data at each node
        width = self.width
        # number of column nodes
        n = int(data.shape[1]/width)
        decoder = general_decoder(G)

        # init nodes and partition data.
        # rows and cols must both be divisible by 7 according to the new scheme.
        # Also, I think that I need to know store the nodes in a more clever way maybe... what a mess
        nodes_list = np.empty(n+1, dtype=object)
        # make an n length list of nodes where each  node stores some slice of the columns of the data
        for i in range(n):
            nodes_list[i] = node(data[:, i*width: width*(i+1)], decoder, G)
        # make the extra node
        nodes_list[-1] = node(data[:, width*n:], decoder, G)
        return nodes_list

    def uniform_query(self, w, lvl, dict, X, index):
        """
        :param w: query, values are from an arithmetic sequence, potentially incomplete
        :param master: stores data array which is being queried
        :return: <data,w> or <w,data>
        """
        if not is_approx_arithmetic_sequence(index):
            raise ValueError(f"recieved bad index{index}")
        w_flat = w.flatten()
        values = np.unique(w_flat)
        d_min = np.min(np.diff(np.sort(values)))  # calculation of minimum difference

        a = np.min(values)
        d = d_min

        if len(w) == X.shape[1]:
            expected_len = 2 ** lvl
            actual = X @ w
            dict['query type'] = ['w']

        else:
            expected_len = 2 ** lvl
            actual = w @ X
            dict['query type'] = ['grd']
        # robust index creation is needed
        """if len(values) != expected_len:
            values = impute(values, expected_len, dict)
        index = values
        index = index.reshape(-1, 1)"""

        # create query table
        column_names = list(range(0, lvl + 1))

        table = np.hstack((index, generate_binary_matrix(lvl)))
        query_table = pd.DataFrame(table, columns=column_names)
        query_table = query_table.set_index(query_table.columns[0])

        # get the correct parity from master
        if w.shape[0] == 1:
            parity = np.hstack([node.row_parity for node in self.nodes_list]).reshape(1, -1)
            assert np.allclose(parity, self.row_parity, atol=1e-5)
        elif w.shape[1] == 1:
            parity = np.sum([node.col_parity for node in self.nodes_list], axis=0)
            # print(np.hstack((parity, Master.col_parity)))
            assert np.allclose(parity, self.col_parity, atol=1e-5)

        # add up responses according to algorithm

        response = parity * (2 * a + (2 ** lvl - 1) * d) / 2

        # Vectorized approach to construct new queries
        for i in range(1, lvl + 1):
            response += ((2 ** (lvl - i - 1)) * d) * self.query(query_table.loc[w_flat, i].values.reshape(w.shape), X,
                                                                  dict)

        # ensure that query is done correctly

        if not np.allclose(response.reshape(-1, 1), actual.reshape(-1, 1), atol=1e-3):
            error = np.linalg.norm(response - actual)
            print("response, actual \n", np.hstack((response.reshape(-1, 1)[0:5], actual.reshape(-1, 1)[0:5])), "\n")
            raise ValueError(f"query does not work: {np.unique(w_flat).reshape(-1, 1)}, with error: {error}")

        return response


"""
data = np.random.rand(14,21)
coded_cols_per_node = 2
G = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, 1, 1, 1, 1],
    [-1, 1, 1, -1, -1, 1, 1],
    [1, -1, -1, -1, -1, 1, 1],
    [1, -1, 1, -1, 1, -1, 1],
    [-1, 1, -1, -1, 1, -1, 1],
    [-1, -1, 1, 1, -1, -1, 1],
    [1, 1, -1, 1, -1, -1, 1]
]).T
Master_test = master(data, None, coded_cols_per_node)
normal_node = Master_test.nodes_list[0]
edge_node = Master_test.nodes_list[1]
print(f" the nodes list looks like: {Master_test.nodes_list.shape},"
      f" the node looks like: {normal_node.data.shape}, \n"
      f" the silly node is like{edge_node.data.shape}")
my_array = np.array([-1, 1])
w = np.random.choice(my_array, size=14, replace=True).reshape(1,-1)

"""