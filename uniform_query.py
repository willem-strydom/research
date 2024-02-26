from query import query
import numpy as np
import pandas as pd

def uniform_query(w, master):
    """
    :param w: query, values are in arithmetic sequence
    :param master: stores data array which is being queried
    :return: <data,w> or <w,data>
    """
    values = sorted(set(w))
    d_min = values[1] - values[0]
    for i in range(1,len(values)):
        d = values[i] - values[i-1]
        if d < d_min:
            d_min = d
    a = np.min(w)
    d = d_min

    # make lookup table and key to map w to {-1,1} querries
    index = [a + i*d for i in range(len(values))]
    first = [-1,-1,1,1]
    second = [-1,1,-1,1]
    table = np.hstack((index,first,second))
    query_table = pd.DataFrame(table)
    query_table = query_table.set_index(query_table.columns[0])

    # construct the new queries and use query() to send them to the master
    w0 = np.array([query_table[w[i],0] for i in range(len(w))]).reshape(w.shape)
    w1 = np.array([query_table[w[i],1] for i in range(len(w))]).reshape(w.shape)
    ans0 = query(w0, master.nodes_array)
    ans1 = query(w1, master.nodes_array)

    # get the corresponding parity depending on shape of w
    if w.shape[0] == 1:
        parity = master.row_parity
    elif w.shape[1] ==1:
        parity = master.col_parity

    # add up responses according to algorithm

    response = d*ans0 + (d/2)*ans1 + ((2*a + 3*d)/2) * parity
    return response
