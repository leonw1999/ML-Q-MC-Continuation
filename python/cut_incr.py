import numpy as np

def cut_incr(M):
    """
    Efficiently constructs a new matrix where:
    - Column 0 = M[:, 0]
    - Column 1 = M[:, 1]
    - Column 2 = M[:, 2] + M[:, 3]
    - Column 3 = M[:, 4] + M[:, 5]
    - ...
    
    Assumes M has at least 2 columns and an even number of columns from index 2 onward.
    """
    if M.shape[1] < 2:
        raise ValueError("Input matrix must have at least two columns.")
    
    rem_cols = M.shape[1] - 2
    if rem_cols % 2 != 0:
        raise ValueError("Number of columns from index 2 onward must be even.")

    # Directly compute paired sums with advanced slicing
    summed_incr = M[:, 2::2] + M[:, 3::2]

    # Use np.concatenate for slightly better perf than hstack here
    return np.concatenate((M[:, :2], summed_incr), axis=1)
