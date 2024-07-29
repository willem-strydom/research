import numpy as np


def is_approx_arithmetic_sequence(seq, tolerance=1e-4):
    # check to make sure that the returned index is approximately an arithmetic sequence
    if len(seq) < 2:
        return False

    # Calculate the common difference
    common_diff = seq[1] - seq[0]

    for i in range(2, len(seq)):
        current_diff = seq[i] - seq[i - 1]
        if abs(current_diff - common_diff) > tolerance:
            print(f"deviation in arith seq: {current_diff, common_diff} on index {i}")
            return False

    return True
def impute(values, expected_len, dict):
    # fill in missing values to create index which can be used to make a lookup table
    dict['imputation'] = [expected_len - len(values)]
    d_min = np.min(np.diff(np.sort(values)))
    a = np.min(values)
    d = d_min
    expected_index = np.linspace(a, a + (expected_len-1) * d, num=expected_len).tolist()
    print(expected_index)
    assert(len(expected_index) == expected_len)
    for val in values:
        # Find the closest number in list2
        closest = min(expected_index, key=lambda x: abs(x - val))
        # Find the index of the closest number
        index = expected_index.index(closest)
        expected_index[index] = val
    if len(expected_index) != expected_len:
        raise ValueError(f" returned {len(expected_index)}, but expected  {expected_len}")
    if not all(elem in expected_index for elem in values):
        raise ValueError(f" returned incomplete list")
    if not is_approx_arithmetic_sequence(expected_index, tolerance=1e-6):
        raise ValueError(f" did not return arithmetic sequence: {expected_index}, from vlaues {values}")

    return np.array(expected_index)

# example bad query which breaks the imputation function