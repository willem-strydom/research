import numpy as np
import pandas as pd
from coded_computation.generate_binary_matrix import generate_binary_matrix
from config import X
from coded_computation.impute import impute

def uniform_query(w, Master, lvl):
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

    if len(w) == X.shape[1]:
        expected_len = 2**lvl
        actual = X @ w

    else:
        expected_len = 2**lvl
        actual = w @ X
    # robust index creation is needed
    if len(values) != expected_len:
        # print(f"correcting bad quantization {q, expected_len, values} \n")
        values = impute(values, expected_len)
    index = values
    index = index.reshape(-1, 1)

    # create query table
    column_names = list(range(0, lvl+1))

    table = np.hstack((index, generate_binary_matrix(lvl)))
    query_table = pd.DataFrame(table, columns=column_names)
    query_table = query_table.set_index(query_table.columns[0])

    # get the correct parity from master
    if w.shape[0] == 1:
        parity = np.hstack([node.row_parity for node in Master.nodes_list]).reshape(1,-1)
        assert np.allclose(parity, Master.row_parity, atol = 1e-5)
    elif w.shape[1] == 1:
        parity = np.sum([node.col_parity for node in Master.nodes_list], axis=0)
        # print(np.hstack((parity, Master.col_parity)))
        assert np.allclose(parity, Master.col_parity, atol = 1e-5)

    # add up responses according to algorithm

    response = parity * (2*a + (2**lvl - 1)*d)/2

    # Vectorized approach to construct new queries
    for i in range(1,lvl+1):
        response += ((2**(lvl-i-1))*d) * Master.query(query_table.loc[w_flat, i].values.reshape(w.shape), X)

    # ensure that query is done correctly

    if not np.allclose(response.reshape(-1,1), actual.reshape(-1,1), atol = 0.01):
        error = np.linalg.norm(response - actual)
        print("response, actual \n", np.hstack((response.reshape(-1,1)[0:5], actual.reshape(-1,1)[0:5])),"\n")
        raise ValueError(f"query does not work: {error}")

    return response
