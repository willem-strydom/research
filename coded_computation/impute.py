import numpy as np

def impute(values, expected_len, dict):
    """
    :param values: arithmetic sequence with missing values
    :param expected_len: expect number of vals
    :return: an ordered sequence of length expected_len
    """
    recieved_vals = values.copy()
    dict['imputation'] = [expected_len - len(recieved_vals)]
    d_min = np.min(np.diff(np.sort(values)))
    a = np.min(values)
    d = d_min
    expected_array = np.arange(a, a + expected_len * d, d)
    for i in range(expected_len - 1):
        if i == len(values) or len(values) == expected_len:
            break
        expected_val = a + d * i
        # impute the missing value.
        if not np.isclose(values[i], expected_val, atol=1e-4):
            values = np.insert(values, i, expected_val)

    if not len(values) == expected_len:
        max_val = np.max(values)
        for i in range(expected_len - len(values)):
            values = np.append(values, max_val + d * (i+1))

    if len(values) != expected_len:
        raise ValueError(f" returned{values}, {len(values)}, but expected  {expected_len}, and received {recieved_vals}")
    if not all(elem in values for elem in recieved_vals):
        raise ValueError(f" returned incomplete list ")
    if not np.allclose(expected_array , values, atol = 1e-4):
        raise ValueError(f" bad index: expected {expected_array}, got {values}")
    return values

"""
vals = [-0.70817232, -0.47939688, -0.42855789, -0.37771891, -0.35229941, -0.31417017,
 -0.28875068, -0.17436296, -0.11081422, -0.07268498, -0.04726549, -0.021846,
  0.04170274,  0.09254173,  0.11796122, 0.21963919,  0.27047818,  0.33402692,
  0.42299514,  0.71531932,  0.75344856, 0.77886805,  0.86783628,  0.90596552]

impute(vals, 128)"""

