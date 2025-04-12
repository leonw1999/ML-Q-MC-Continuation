import numpy as np

def cut_paths(M):
    """
    Removes the 3rd, 5th, 7th, ... rows (0-based indices 2, 4, 6, ...) from the matrix M.

    Parameters:
    - M: np.ndarray, input matrix

    Returns:
    - np.ndarray, matrix with selected rows removed
    """
    
    mask = np.ones(M.shape[0], dtype=bool)
    mask[2::2] = False 
    return M[mask]