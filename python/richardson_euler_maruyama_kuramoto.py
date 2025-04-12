import numpy as np
from cut_incr import cut_incr
from euler_maruyama_kuramoto import euler_maruyama_kuramoto
from obj_fctn import obj_fctn

def richardson_euler_maruyama_kuramoto(M_fine, T, K, sigma, P):
    """
    Performs Richardson Extrapolated Euler-Maruyama method for the Kuramoto system using
    the coarse and fine grids.

    Parameters:
    - M_fine: (M_fine * P, 2 + N_fine) array, fine matrix
    - T: float, final time
    - K: float, coupling strength
    - sigma: float, noise intensity
    - P: int, number of oscillators per sample

    Returns:
    - X_rich: (M, P) array, final phase angles at time T for each sample after Richardson extrapolation
    """
    # Coarse and fine time step sizes
    M_coarse = cut_incr(M_fine)

    # Run Euler-Maruyama method for coarse grid
    X_coarse = obj_fctn(euler_maruyama_kuramoto(M_coarse, T, K, sigma, P), P)
    
    # Run Euler-Maruyama method for fine grid
    X_fine = obj_fctn(euler_maruyama_kuramoto(M_fine, T, K, sigma, P), P)

    # Richardson Extrapolation
    X_rich = 2 * X_fine - X_coarse

    return X_rich

