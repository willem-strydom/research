# node class
#take slices instead of dot product
import numpy as np
class node:
    def __init__(self, data: np.ndarray, decoder: dict, G: np.ndarray):
        """
        :param data: data, new constraint, data is 7X7 matrix. will not work otherwise
        :param decoder: lookup table maybe?
        :param G: encoding matrix
        """
        self.G = G
        self.decoder = decoder
        self.eye = data
        self.row_par = np.dot(data.T,G)[:,G.shape[0]:]
        self.col_par = np.dot(data,G)[:,G.shape[0]:]

    def query(self, w:np.ndarray):
        """
        compute data@w, or w@data depending on shape of w, either 7x1 or 1x7, respectively
        :param w: query on raw data
        :return: self.data @ w , or w@self.data and number of cols of coded data accessed to do that computation
        """
        # return data@w.T based on decoding protocol...
        #print(f"hi, {w} --\n")
        low_acc_w = self.decoder[tuple(w.flatten())]
        index = np.where(low_acc_w != 0, 1, 0)
        access = np.sum(index)
        index = index.astype(bool)

        # slice the accessed columns and then take the dot product
        # according to whether row or cols are being accessed
        # accessed_columns = self.coded_data[:,index] was the original method

        #columns
        m = self.G.shape[0]
        if w.shape[1] == 1:

            accessed_columns = np.hstack((self.eye[:, index[0:m]], self.col_par[:,index[m:]]))
            nonzero_low_acc_w = low_acc_w[index]
            response = accessed_columns @ nonzero_low_acc_w

        #rows
        if w.shape[0] == 1:

            accessed_columns = np.hstack((self.eye.T[:, index[0: m]], self.row_par[:, index[m:]]))
            nonzero_low_acc_w = low_acc_w[index]
            response = accessed_columns @ nonzero_low_acc_w


        return response
# test for making parity check matrix
"""data = np.random.rand(2,2)
decoder = {}
B = np.array([
    [1,0,1],
    [0,1,1]
])
test_node = node(data,decoder,B)
print(test_node.H)"""
# test is query is working alright
"""data = np.random.rand(2,2)

B = np.array([[1,0,1], #parity code... satisfies the closed under compliment bit I think
              [0,1,1]
              ])

decoder = {
    (-1,-1): np.array([0,0,-1]),
    (-1,1):np.array([-1,1,0]),
    (1,-1):np.array([1,-1,0]),
    (1,1):np.array([0,0,1])
}# Best guess as to what the lookup table should look like...
node = node(data,decoder,B)
w = np.array([1,-1])
print(node.query(w))
print(data @ w.T)"""