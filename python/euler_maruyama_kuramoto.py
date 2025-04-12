import numpy as np

def euler_maruyama_kuramoto(M_matrix, T, K, sigma, P):
    """
    Simulates the McKean-Vlasov Kuramoto system using the Euler-Maruyama method
    with inputs consolidated into a single matrix.

    Parameters:
    - M_matrix: (M*P, 2+N) array
        First column: initial phase angles (flattened)
        Second column: natural frequencies (flattened)
        Next N columns: Wiener paths (flattened)
    - T: float, final time
    - K: float, coupling strength
    - sigma: float, noise intensity
    - P: int, number of oscillators per sample

    Returns:
    - X: (M, P) array, final phase angles at time T for each sample
    """
    total_rows, total_cols = M_matrix.shape
    N = total_cols - 2  # number of time steps

    # Check that total rows is divisible by P
    if total_rows % P != 0:
        raise ValueError(f"The number of rows in M_matrix ({total_rows}) must be divisible by P ({P}).")

    M = total_rows // P  # number of samples
    dt = T / N

    # Extract components
    X0 = M_matrix[:, 0].reshape((M, P))
    nu = M_matrix[:, 1].reshape((M, P))
    dW = M_matrix[:, 2:].reshape((M, P, N))

    # Initialize
    X = np.copy(X0)

    # Euler-Maruyama integration
    for n in range(N):
        mean_field = np.mean(np.sin(X[:, None, :] - X[:, :, None]), axis=2)
        X += (nu + (K / 2) * mean_field) * dt + sigma * np.sqrt(dt) * dW[:, :, n]

    return X
