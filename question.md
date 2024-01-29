# questions about what I am supposed to be doing \n
1. what is supposed to be stored at each node exactly. Right now the node attributes are the generator matrix used to code the
data, the coded data, and the lookup table dictionary to decode.
2. When decoding, we give a node the query w, then the use the lookup table to find the "low access" version. Is it then
sufficient to just take and return the dot product np.dot(w_low_access,self.coded_data), or is there a particular way
that the dot product must be computed in order to get low access.
3. Specifics of the master routine. Currently, I have a "master" function which inits m nodes, then I have a seperate
"query" function which partitions, sends, and combines a query to the nodes initialized by the master