# node class
#take slices instead of dot product
import numpy as np
class node:
    def __init__(self, data: np.ndarray, decoder: dict, G: np.ndarray):
        """
        :param data: data
        :param decoder: lookup table maybe?
        :param G: encoding matrix
        """
        self.G = G
        self.decoder = decoder
        self.coded_data = np.dot(data,G)

    def query(self, w:np.ndarray):
        # return data@w.T based on decoding protocol...
        low_acc_w = self.decoder[tuple(w)] #python dict need immutable keys I just learned
        index = np.where(low_acc_w!=0,1,0)
        access = np.sum(index)
        index = index.astype(bool)
        # response = self.coded_data @ low_acc_w.T
        accessed_columns = self.coded_data[:,index]
        nonzero_low_acc_w = low_acc_w[index]
        response = accessed_columns @ nonzero_low_acc_w

        return response, access
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