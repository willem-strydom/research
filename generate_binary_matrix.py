import numpy as np
"""
make a numpy array of size 2^n x n where each row is the binary representation of the row number in {-1,1}.
(-1 mapped to 0)
"""

def generate_binary_matrix(n):
    # Number of rows in the output matrix
    num_rows = 2 ** n

    # Generate all integers from 0 to 2^n - 1
    integers = np.arange(num_rows)

    # Convert integers to binary, strip the '0b' prefix, and pad with zeros to ensure n bits
    binary_strings = [bin(x)[2:].zfill(n) for x in integers]

    # Convert binary strings to a list of lists of integers
    binary_matrix = np.array([list(map(int, x)) for x in binary_strings])

    binary_matrix = np.where(binary_matrix == 0, -1, 1)

    return binary_matrix

