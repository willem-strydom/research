from query import query
import numpy as np
import pandas as pd
from generate_binary_matrix import generate_binary_matrix
import math
from config import X, w_lvl, grd_lvl

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
    index = a + np.arange(len(values)) * d
    tolerance = 0.001
    if not np.allclose(index, values, atol=tolerance):
        # robust index creation is needed since there is a decent chance that at some point
        # a non-representative w will be passed
        raise ValueError(f"bad quantization received: {values}, {index}, {d_min}")
    index = values # check that all vals are present in query
    index = index.reshape(-1, 1)

    # create query table
    q = len(values)
    lvl = int(math.log2(q))
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
