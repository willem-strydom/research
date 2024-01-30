import numpy as np
def query(w1, nodes_array):
    """
    :param w: query on RAW data
    :param nodes_array: the storage nodes from master
    :return: np.dot(w,data) but like with low access
    """
    m = len(nodes_array)
    w = w1.flatten()
    width = nodes_array[0].G.shape[0] # size of raw data at each node is needed to partition w. assume systematic B
    # partition w and do a query
    access = np.zeros(m)
    ans_array, access[0] = nodes_array[0].query(w[width*0:width*(0+1)])

    for i in range(1,m):
        query = nodes_array[i].query(w[width*i:width*(i+1)])
        ans_array = np.vstack((query[0]))
        access[i] = (query[1])
    return np.sum(ans_array, axis = 1), np.sum(access) # add the responses together, I think sum them as columns is axis = 1 ...
