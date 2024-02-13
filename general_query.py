from query import query
import numpy as np

def general_query(w, master):
    """
    :param w: for now, a query in {0,1}^n
    :param type: row or column, whether the query is on the rows of cols of data
    :param nodes_array: the coded data array
    :return: data @ w basically
    """

    # I think that the general scheme is map 0 -> -1 and then add the new query to the parity which is stored generally
    w1 = np.where(w == 0, -1,1)
    nodes_array = master.nodes_array
    parity = 0
    if w.shape[0] == 1:
        parity = master.row_parity
    else:
        parity = master.col_parity
    return (query(w1,nodes_array) + parity) /2
