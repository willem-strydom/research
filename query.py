import numpy as np
def query(w, nodes_array):
    """
    :param w: query on RAW data
    :param nodes_array: the storage nodes from master in an mxn array
    :return: np.dot(w,data) but like with low access
    """
    m = nodes_array.shape[0]
    n = nodes_array.shape[1]

    width = nodes_array[0, 0].G.shape[0] # size of raw data at each node is needed to partition w. assume systematic B
    # partition w and do a query
    # access = np.zeros(m) im not going to bother computing access, its annoying and we already saw that its about 2 per node
    # ans_array, access[0] = nodes_array[0].query(w[width*0:width*(0+1)])


    ans_array = []
    # check w is a querry on the rows
    if w.shape[0] == 1:
        w = w.flatten()  # need the shape so that the query is performed right
        for m in range(m):
            response = np.array([])
            for n in range(n):
                response.append(nodes_array[m,n].querry(w[width*m:width*(1+m)]))
            response = np.hstack((response))
            ans_array.append(response)
        ans_array = np.vstack((ans_array))

    # for a querry on the columns... I think just change the order of the loops maybe
    if w.shape[1] == 1:
        w = w.flatten()  # need the shape so that the query is performed right
        for n in range(n):
            response = np.array([])
            for m in range(m):
                response.append(nodes_array[m,n].querry(w[width*m:width*(1+m)]))
            response = np.hstack((response))
            ans_array.append(response)
        ans_array = np.vstack((ans_array))





    """for i in range(1,m):
        query = nodes_array[i].query(w[width*i:width*(i+1)])
        ans_array = np.vstack((query[0]))
        access[i] = (query[1])"""
    return np.sum(ans_array, axis = 1)
