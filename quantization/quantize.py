import numpy as np
from quantization.binning import binning
from coded_computation.impute import is_approx_arithmetic_sequence

def quantize(vals, level, type):
    """
    :param vals: numpy nx1 or 1xn array of func arguments that will be quantized
    :param level: int: log2 number of bins/quantization levels
    :return: nx1 numpy array of quantized values
    """

    partitions = binning(vals, level, type)
    # alpha is a list of which bin each val belongs to

    # processing partitions... maybe not a great way to do this
    step = partitions[1] - partitions[0]
    # remove edge partitions since it should j be +- \infty
    partitions = partitions[1:-1]
    # index for which bin each respective value falls into
    alpha = np.digitize(vals, partitions).flatten()
    # map them to appropriate values based on the mean of func evaluation of the respective bin edges
    beta = np.zeros(alpha.shape)
    arith_seq = [partitions[0] - step/2]
    for i in range(1, len(partitions)):
        arith_seq.append(
            (partitions[i] + partitions[i-1])/2
                          )
    arith_seq.append(
        partitions[-1] + step/2
    )
    i = 0

    for a in alpha:
        beta[i] = arith_seq[a]
        i += 1
    assert is_approx_arithmetic_sequence(arith_seq)
    return beta.reshape(vals.shape), np.array(arith_seq).reshape(-1,1)
# some checks and testing
vals = np.random.normal(0,4,200)
lvl = 9
type = "unif"
result, seq = quantize(vals, lvl, type)
print(f"number of unique quantizations: {len(np.unique(result))} \n")
print(f"range of quantized vals {np.min(result)}, {np.max(result)}")
print(f"range of unquantized vals {np.min(vals)}, {np.max(vals)}")
print(f"mean abs err: {np.mean(np.abs(vals - result))}")
feasible = (np.max(result) - np.min(result))/ 2**lvl
print(f"feasible error: {feasible}")
print(f" number of bad quantizations {np.sum(np.abs(vals - result) > feasible)}")
print(f"median err: {np.median(np.abs(vals - result))}")

set1 = set(result)
set2 = set(seq.flatten())
print(set1.issubset(set2))

