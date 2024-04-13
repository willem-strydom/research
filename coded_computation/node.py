# node class

import numpy as np


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
        if data.shape[0] % G.shape[0] != 0 or data.shape[1] % G.shape[0] != 0:
            raise ValueError("Data dimensions must be divisible by G.shape[0]")

        # Compute and store parities in separate matrices
        self.row_parities_matrix, self.col_parities_matrix = self.compute_parities()

    def compute_parities(self):
        rows, cols = self.data.shape
        chunk_size = self.G.shape[0]  # Assuming G is square
        num_chunks_row = rows // chunk_size
        num_chunks_col = cols // chunk_size

        # Initialize matrices to store row and column parities for each chunk
        row_parities_matrix = np.empty((num_chunks_row, num_chunks_col), dtype=object)
        col_parities_matrix = np.empty((num_chunks_row, num_chunks_col), dtype=object)

        for i in range(num_chunks_row):
            for j in range(num_chunks_col):
                # Extract the current chunk
                chunk = self.data[i * chunk_size:(i + 1) * chunk_size, j * chunk_size:(j + 1) * chunk_size]

                # Compute row and column parities for the chunk
                row_parities = chunk @ self.G
                col_parities = chunk.T @ self.G

                # Store the computed parities in their respective matrices
                row_parities_matrix[i, j] = row_parities
                col_parities_matrix[i, j] = col_parities

        return row_parities_matrix, col_parities_matrix

    def query(self, w: np.ndarray):
        """
        Compute data@w, or w@data depending on shape of w, either dx1 or 1xd, respectively.
        The method has been updated to work with the new class structure where each
        chunk's row and column parities are stored separately.
        :param w: query vector
        :return: Result of the query and number of chunks accessed.
        """
        # Determine the size of each chunk
        chunk_size = self.G.shape[0]

        # Determine query type (row vs column) and process accordingly
        if w.shape[1] == 1:  # col
            column_response_accumulator = None
            for j in range(self.col_parities_matrix.shape[1]):
                row_response_accumulator = []
                for i in range(self.col_parities_matrix.shape[0]):
                    low_access_w = self.decoder[tuple(w[chunk_size * i:chunk_size * (i + 1),:].flatten())]
                    bool_index = np.abs(low_access_w).astype(bool)
                    chunk_slice = np.hstack(
                        (self.data[chunk_size * i:chunk_size * (i + 1), chunk_size * j:chunk_size * (j + 1)]
                         , self.col_parities_matrix[i, j]))
                    nonzero_low_acc_w = low_access_w[bool_index]
                    accessed_columns = chunk_slice[:, bool_index]
                    chunk_response = accessed_columns @ nonzero_low_acc_w
                    print(chunk_response)
                    print(self.data[chunk_size * i:chunk_size * (i + 1), chunk_size * j:chunk_size * (j + 1)] @ w[chunk_size * i:chunk_size * (i + 1),:])

                    row_response_accumulator.extend(chunk_response)
                row_response_accumulator = np.array(row_response_accumulator)
                if column_response_accumulator is None:
                    column_response_accumulator = row_response_accumulator
                else:
                    column_response_accumulator += row_response_accumulator

            response = np.array(column_response_accumulator).reshape(-1, 1)

        elif w.shape[0] == 1:  # row
            column_response_accumulator = None
            for i in range(self.row_parities_matrix.shape[0]):
                row_response_accumulator = []
                for j in range(self.row_parities_matrix.shape[1]):
                    low_access_w = self.decoder[tuple(w[:,chunk_size * i:chunk_size * (i+1)].flatten())]
                    bool_index = np.abs(low_access_w).astype(bool)
                    chunk_slice = np.hstack(
                        (self.data[chunk_size * i:chunk_size * (i + 1), chunk_size * j:chunk_size * (j + 1)].T
                         , self.row_parities_matrix[i, j]))
                    nonzero_low_acc_w = low_access_w[bool_index]
                    accessed_columns = chunk_slice[:,bool_index]
                    chunk_response = accessed_columns @ nonzero_low_acc_w
                    row_response_accumulator.extend(chunk_response)
                row_response_accumulator = np.array(row_response_accumulator)
                print(row_response_accumulator)
                if column_response_accumulator is None:
                    column_response_accumulator = row_response_accumulator
                else:
                    column_response_accumulator += row_response_accumulator


            response = np.array(column_response_accumulator).reshape(1, -1)
        return response

# test is query is working alright

data = np.random.rand(6,8)

B = np.array([[1], #parity code... satisfies the closed under compliment bit I think
              [1]
              ])

decoder = {
    (-1,-1): np.array([0,0,-1]),
    (-1,1):np.array([-1,1,0]),
    (1,-1):np.array([1,-1,0]),
    (1,1):np.array([0,0,1])
}# Best guess as to what the lookup table should look like...
node = node(data,decoder,B)
w = np.array([[-1, -1, 1, 1, 1, 1,1,-1]]).T
print(node.query(w))
print(data@w)

# also measure the access
# also do a distributed storage system which is uncoded
# compare in terms of time and performance