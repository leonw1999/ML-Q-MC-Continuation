import numpy as np

def euler_maruyama_kuramoto(X0, nu, dW, T, K, sigma):
    """
    Simulates the McKean-Vlasov Kuramoto system using the Euler-Maruyama method.

    Parameters:
    - X0: (S, N) array, initial phase angles for S samples of N oscillators
    - nu: (S, N) array, natural frequencies for S samples of N oscillators
    - dW: (S, N, M) array, Wiener increments for S samples over M time steps
    - T: float, final time
    - K: float, coupling strength
    - sigma: float, noise intensity

    Returns:
    - X_T: (S, N) array, final phase angles at time T for each sample
    """
    S, N = X0.shape  # Number of samples, number of oscillators
    M = dW.shape[2]  # Number of time steps
    dt = T / M       # Time step size

    # Initialize phase angles for all samples
    X = np.copy(X0)

    for m in range(M):
        # Compute empirical mean-field interaction for each sample
        mean_field = np.mean(np.sin(X[:, None, :] - X[:, :, None]), axis=2)

        # Euler-Maruyama step for all samples
        X += (nu + (K / 2) * mean_field) * dt + sigma * np.sqrt(dt) * dW[:, :, m]

    return X  # Final phase angles for all samples