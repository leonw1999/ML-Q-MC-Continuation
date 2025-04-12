import numpy as np

def cut_pts(M):
    """
    Splits matrix M into two matrices:
    - M1: rows 0, 2, 4, 6, ...
    - M2: rows 1, 3, 5, 7, ...

    Parameters:
    - M: np.ndarray, input matrix

    Returns:
    - M1, M2: np.ndarray, alternating row splits of M
    """


    M1 = M[::2]  # even-indexed rows
    M2 = M[1::2]  # odd-indexed rows

    return M1, M2
