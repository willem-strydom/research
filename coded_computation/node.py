# node class

import numpy as np
import csv

class node:
    """
    Simulate a coded storage server. It insane... It got way out of hand.
    need to:
    1. parse data up into codeable chunks, encode (not so bad)
    2. get a query in {+-1}^d, figure out if row of col query, correctly slice, distribute, compute and combine

    """
    def __init__(self, data: np.ndarray, decoder: dict, G: np.ndarray):
        """
        Initialize the node with data, a decoder, and an encoding matrix G.
        Data's dimensions must be divisible by those of G.
        G is just the non-systematic part of a give code btw
        """
        self.G = G
        self.decoder = decoder
        self.data = data
        if G is not None:
            if data.shape[0] % G.shape[0] != 0 or data.shape[1] % G.shape[0] != 0:
                raise ValueError("Data dimensions must be divisible by G.shape[0]")

        # Compute and store parities in separate matrices
        self.row_parities_matrix, self.col_parities_matrix = self.compute_parities()
        self.row_parity = np.ones((1, self.data.shape[0]))@self.data
        self.col_parity = self.data@np.ones((self.data.shape[1], 1))

    def compute_parities(self):
        # procedure if there is no encoding
        if self.decoder == None:
            return None, None
        rows, cols = self.data.shape
        slice_size = self.G.shape[0]
        num_slices_row = rows // slice_size
        num_slices_col = cols // slice_size

        # Initialize matrices to store row and column parities for each chunk
        row_parities_matrix = np.empty(num_slices_row, dtype=object)
        col_parities_matrix = np.empty(num_slices_col, dtype=object)

        # row parities
        for i in range(num_slices_row):
            row_parities_matrix[i] = self.data[slice_size*i: slice_size*(i+1),:].T@self.G
        # col parities
        for j in range(num_slices_col):
            col_parities_matrix[j] = self.data[:,slice_size*j: slice_size*(j+1)]@self.G

        return row_parities_matrix, col_parities_matrix

    def query(self, w, dict):
        """
        Compute data@w, or w@data depending on shape of w, either dx1 or 1xd, respectively.
        The method has been updated to work with the new class structure where each
        chunk's row and column parities are stored separately.
        :param w: query vector
        :return: Result of the query and number of chunks accessed.
        """
        # add a line to do the un-coded version?
        if self.decoder == None:
            # just do the dot product
            if w.shape[0] == 1:
                return (w@self.data).reshape(1,-1)
            else:
                return (self.data@w).reshape(-1,1)


        # Determine the size of each chunk
        chunk_size = self.G.shape[0]
        # partition w, for each partition use the low acc scheme to do the dot product
        if w.shape[0] == 1:
            response = np.zeros((1,self.data.shape[1]))
            num_partitions = int(w.shape[1]/chunk_size)
            w_partitions = [w[:, chunk_size*i:chunk_size*(i+1)] for i in range(num_partitions)]
            for i, w_star in enumerate(w_partitions):
                data_slice = self.data[chunk_size*i:chunk_size*(i+1),:].T
                parities_slice = self.row_parities_matrix[i]
                coded_data = np.hstack((data_slice,parities_slice))
                low_access_w = self.decoder[tuple(w_star.flatten())]
                boolean_w = np.where(low_access_w != 0 , True, False)
                accessed_slices = coded_data[:,boolean_w]
                accessed_w = low_access_w[boolean_w]
                dict['access'] += len(accessed_w)
                inner_prod = (accessed_slices @ accessed_w).reshape(response.shape)
                response += inner_prod
            return response
        # if column
        if w.shape[1] == 1:
            response = np.zeros((self.data.shape[0],1))
            num_partitions = int(w.shape[0]/chunk_size)
            w_partitions = [w[chunk_size*i:chunk_size*(i+1),:] for i in range(num_partitions)]
            for i, w_star in enumerate(w_partitions):
                data_slice = self.data[:, chunk_size*i:chunk_size*(i+1)]
                parities_slice = self.col_parities_matrix[i]
                coded_data = np.hstack((data_slice,parities_slice))
                low_access_w = self.decoder[tuple(w_star.flatten())]
                boolean_w = np.where(low_access_w != 0 , True, False)
                accessed_slices = coded_data[:,boolean_w]
                accessed_w = low_access_w[boolean_w]
                dict['access'] += len(accessed_w)
                inner_prod = (accessed_slices @ accessed_w).reshape(response.shape)
                response += inner_prod
            return response








# test is query is working alright

"""data = np.random.rand(6,8)

B = np.array([[1], #parity code... satisfies the closed under compliment bit I think
              [1]
              ])

decoder = {
    (-1,-1): np.array([0,0,-1]),
    (-1,1):np.array([-1,1,0]),
    (1,-1):np.array([1,-1,0]),
    (1,1):np.array([0,0,1])
}
# Best guess as to what the lookup table should look like...
test_node = node(data,decoder,B)
print(f"this is the row parity matrix {test_node.row_parities_matrix[0].shape} \n "
      f"this is the col parity matrix {test_node.col_parities_matrix[0].shape} \n "
      f"this is the row parity bit {test_node.row_parity} \n"
      f"this is the col parity bit {test_node.col_parity} \n")
w = np.array([[-1, -1, 1, 1, 1, 1]])
print(test_node.query(w))
print(w@data)"""


# also measure the access
# also do a distributed storage system which is uncoded
# compare in terms of time and performance