from query import query
import numpy as np
import pandas as pd
from generate_binary_matrix import generate_binary_matrix
import math
from config import X, w_lvl, grd_lvl
from impute import impute

def uniform_query(w, Master):
    """
    :param w: query, values are in arithmetic sequence
    :param master: stores data array which is being queried
    :return: <data,w> or <w,data>
    """

    w_flat = w.flatten()
    values = np.unique(w_flat)  # More efficient and readable way to get unique values
    d_min = np.min(np.diff(np.sort(values)))  # More efficient calculation of minimum difference

    a = np.min(values)
    d = d_min

    # Construct index for lookup table
    q = len(values)
    lvl = 0
    expected_len = 0
    if w.shape[0] == 1:
        expected_len = 2**w_lvl
        lvl = w_lvl
    else:
        expected_len = 2**grd_lvl
        lvl = grd_lvl

    if q != expected_len:
        print(f"correcting bad quantization {q, expected_len, values} \n")
        # robust index creation is needed since there is a decent chance that at some point
        # a non-representative w will be passed
        # search for missing values, and impute
        values = impute(values,expected_len)
    else:
        print(f"good quantization")
    index = values
    index = index.reshape(-1, 1)

    # create query table
    column_names = list(range(0, lvl+1))

    table = np.hstack((index, generate_binary_matrix(lvl)))
    query_table = pd.DataFrame(table, columns=column_names)
    query_table = query_table.set_index(query_table.columns[0])

    # get the correct parity from master
    if w.shape[0] == 1:
        parity = Master.row_parity
    elif w.shape[1] == 1:
        parity = Master.col_parity

    # add up responses according to algorithm

    response = parity * (2*a + (2**lvl - 1)*d)/2

    # Vectorized approach to construct new queries
    for i in range(1,lvl+1):
        response += ((2**(lvl-i-1))*d) * query(query_table.loc[w_flat, i].values.reshape(w.shape), Master, X)

    # ensure that query is done correctly
    actual = 0
    if w.shape[0] == 1:
        actual = w@X
    if w.shape[1] == 1:
        actual = X@w
    if not np.allclose(response.reshape(-1,1), actual.reshape(-1,1), atol = 0.01):
        error = np.linalg.norm(response - actual)
        print("response, actual", np.hstack((response.reshape(-1,1)[0:5], actual.reshape(-1,1)[0:5])),"\n")
        raise ValueError(f"query does not work: {error}")

    return response
