import numpy as np

def query(w, Master, X):
    """
    :param w: query on RAW data in {-1,1}^n
    :param nodes_array: the storage nodes from master in an mxn array
    :return: np.dot(w,data) but like with low access
    """
    nodes_array = Master.nodes_array
    rows = nodes_array.shape[0]
    cols = nodes_array.shape[1]

    width = nodes_array[0, 0].G.shape[0] # size of raw data at each node is needed to partition w. assume systematic B
    # partition w and do a query
    # access = np.zeros(m) im not going to bother computing access, its annoying and we already saw that its about 2 per node
    # ans_array, access[0] = nodes_array[0].query(w[width*0:width*(0+1)])


    ans_array = []
    # check w is a querry on the rows
    if w.shape[0] == 1:
        w = w.flatten()  # need the shape so that the query is performed right
        for m in range(rows):
            response = []
            for n in range(cols):
                response.append(nodes_array[m,n].query(w[width*m:width*(1+m)].reshape(1,-1)))
            response = np.hstack((response))
            ans_array.append(response)
        ans_array = np.vstack((ans_array))

        resp = np.sum(ans_array, axis=0)
        return resp.reshape(1,-1)

    # for a querry on the columns... I think just change the order of the loops maybe
    if w.shape[1] == 1:
        w = w.flatten()  # need the shape so that the query is performed right
        for n in range(cols):
            response = []
            for m in range(rows):
                response.append(nodes_array[m,n].query(w[width*n:width*(1+n)].reshape(-1,1)))
            response = np.hstack(response)
            ans_array.append(response)
        ans_array = np.vstack(ans_array)

        resp = np.sum(ans_array, axis = 0)
        return resp.reshape(-1,1)

    """for i in range(1,m):
        query = nodes_array[i].query(w[width*i:width*(i+1)])
        ans_array = np.vstack((query[0]))
        access[i] = (query[1])"""

