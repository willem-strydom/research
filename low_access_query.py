import numpy as np
def query(w, nodes_array,n):
    """
    :param w: query on RAW data
    :param nodes_array: the storage nodes from master
    :return: np.dot(w,data) but like with low access
    """
    m = len(nodes_array)
    width = nodes_array[0].G.shape[0] # size of raw data at each node is needed to partition w. assume systematic B
    # partition w and do a query
    ans_array = np.zeros((n,m))
    access = np.zeros(m)
    for i in range(m):
        ans_array[:,i], access[i] = nodes_array[i].query(w[width*i:width*(i+1)])
    return np.sum(ans_array, axis = 1), np.sum(access) # add the responses together, I think sum them as columns is axis = 1 ...
