import numpy as np

def richardson_euler_maruyama_kuramoto(X0, nu, dW, T, K, sigma):
    """
    Implements the Euler-Maruyama method with Richardson extrapolation 
    for the McKean-Vlasov Kuramoto SDE.

    Parameters:
    - X0: (M, P) array, initial phase angles for M samples of P oscillators
    - nu: (M, P) array, natural frequencies for M samples of P oscillators
    - dW: (M, P, 2N) array, Wiener increments for M samples over 2N fine time steps
    - T: float, final time
    - K: float, coupling strength
    - sigma: float, noise intensity

    Returns:
    - X_T: (M, P) array, final phase angles at time T using Richardson extrapolation
    """
    M, P = X0.shape  # Number of samples, number of oscillators
    N = dW.shape[2] // 2  # Number of coarse time steps
    dt = T / N        # Coarse step size
    dt_half = dt / 2  # Fine step size

    # Initialize phase angles
    X_coarse = np.copy(X0)
    X_fine = np.copy(X0)

    # Coarse time stepping (step size dt)
    for n in range(N):
        mean_field = np.mean(np.sin(X_coarse[:, None, :] - X_coarse[:, :, None]), axis=2)
        X_coarse += (nu + (K / 2) * mean_field) * dt + sigma * np.sqrt(dt) * dW[:, :, 2*n]

    # Fine time stepping (step size dt/2)
    for n in range(2 * N):
        mean_field = np.mean(np.sin(X_fine[:, None, :] - X_fine[:, :, None]), axis=2)
        X_fine += (nu + (K / 2) * mean_field) * dt_half + sigma * np.sqrt(dt_half) * dW[:, :, n]

    # Richardson extrapolation: X_T = 2 * X_fine - X_coarse
    X_T = 2 * X_fine - X_coarse

    return X_T
