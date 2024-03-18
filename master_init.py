from master import master
import numpy as np
from config import X
# Initialize the global variable to None
_master_instance = None

def get_or_create_master(*args, **kwargs):
    global _master_instance
    if _master_instance is None:
        # Initialize Master here or use args and kwargs as needed
        I = np.eye(7)
        B = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, 1, 1, 1, 1],
            [-1, 1, 1, -1, -1, 1, 1],
            [1, -1, -1, -1, -1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1],
            [-1, 1, -1, -1, 1, -1, 1],
            [-1, -1, 1, 1, -1, -1, 1],
            [1, 1, -1, 1, -1, -1, 1]
        ]).T
        G = np.hstack((I, B))
        _master_instance = master(X, G)
    return _master_instance

def initialize_master(*args, **kwargs):
    """
    Explicitly initialize the Master instance.
    This can be used to reset or re-initialize the instance with new parameters.
    """
    global _master_instance
    _master_instance = master(*args, **kwargs)
