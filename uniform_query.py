from query import query
import numpy as np
import pandas as pd

def uniform_query(w, master):
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
        print("bad quantization")
    index = values

    q = len(values)
    column_names = list(range(q))
    query_table = pd.DataFrame(table, columns=column_names).set_index('Index')

    # Vectorized approach to construct new queries
    response = np.zeros(w.shape)
    for i in range(1,q+1):
        response += 2**(q-i-1)*d*query_table.loc[w_flat, i].values.reshape(w.shape)

    # get the corresponding parity depending on shape of w
    if w.shape[0] == 1:
        parity = master.row_parity
    elif w.shape[1] == 1:
        parity = master.col_parity

    # add up responses according to algorithm

    response += parity

    return response

q = 3
table = np.array(np.meshgrid(*[[-1, 1]] * q)).reshape(-1, q)
print(table)