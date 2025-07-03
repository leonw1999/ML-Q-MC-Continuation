import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from QMC_sampler import QMC_sampler
from pts_to_paths import pts_to_paths
from euler_maruyama_kuramoto import euler_maruyama_kuramoto

# Assuming the previous classes and functions (QMC_sampler, pts_to_paths, euler_maruyama_kuramoto) are already defined

def compute_residual(previous_X, current_X):
    """
    Compute the residual between the previous and current discretization.
    Residual is the difference between h and 2h discretizations.
    """
    residual = np.abs(current_X - previous_X)
    return np.mean(residual)

def main():
    # Parameters
    T = 10.0  # Final time
    K = 1.0   # Coupling strength
    sigma = 0.1    # Noise intensity
    num_shifts = 200  # Number of random shifts
    max_level = 8  # Max number of levels to simulate
    P_list = []

    # Initialize QMC sampler and generate points
    qmc_sampler = QMC_sampler()
    qmc_sampler.initialize_from_file('genvec.txt')

    # Generate a list to store residuals for each level
    residuals = []

    # Initialize previous_X as None for the first iteration
    previous_X = None

    for level in range(1, max_level + 1):
        # Add level to the QMC sampler and get the points at this level
        qmc_sampler.add_level()
        points = qmc_sampler.points[-1]  # The points from the latest level

        # The shift length is equal to the number of columns in the current level points
        shift_length = points.shape[1]  # The shift length corresponds to the dimension of the points (n)
        P = shift_length - 2

        # Generate uniformly distributed random shifts of size (100, shift_length)
        shifts = np.random.rand(num_shifts, shift_length)

        # Generate paths using pts_to_paths
        paths = pts_to_paths(points, shifts, T)

        # Compute Euler-Maruyama discretization for the current level
        X_current = np.mean(np.cos(euler_maruyama_kuramoto(paths, T, K, sigma, P)))

        # If it's not the first level, compute the residual
        if previous_X is not None:
            residual = compute_residual(previous_X, X_current)
            residuals.append(residual)
            P_list.append(P)

        # Update previous_X for the next iteration
        previous_X = X_current

    # Plot residual vs. levels with log scale on y-axis and normal scale on x-axis
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_level + 1), residuals, label='Residuals', marker='o')
    plt.plot(range(2, max_level + 1), residuals[1] * np.array(P_list)[1]**2 / np.array(P_list)**2, label='O(P^2)')
    plt.yscale('log')
    plt.xlabel('Number of Levels')
    plt.ylabel('Residual (log scale)')
    plt.title('Euler-Maruyama Residuals vs. Levels')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
