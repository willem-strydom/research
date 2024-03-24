import numpy as np

def impute(values, expected_len):
    """
    :param values: arithmetic sequence with missing values
    :param expected_len: expect number of vals
    :return: an ordered sequence of length expected_len
    """
    d_min = np.min(np.diff(np.sort(values)))
    a = np.min(values)
    d = d_min
    for i in range(expected_len - 1):
        if i == len(values):
            break
        expected_val = a + d * i
        # impute the missing value.
        if not np.isclose(values[i], expected_val, atol=1e-4):
            values = np.insert(values, i, expected_val)
    if not len(values) == expected_len:
        max_val = np.max(values)
        for i in range(expected_len - len(values)):
            values = np.append(values, np.array(max_val + d * (i+1)))

    if len(values) != expected_len:
        raise ValueError(f" returned{values}, but expect  {expected_len}")
    return values
