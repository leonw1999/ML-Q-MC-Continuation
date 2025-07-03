import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from QMC_sampler import QMC_sampler
from pts_to_paths import pts_to_paths
from cut_pts import cut_pts
from cut_incr import cut_incr
from euler_maruyama_kuramoto import euler_maruyama_kuramoto
from obj_fctn import obj_fctn

# --- Parameters ---
T = 1.0
K = 1.5
sigma = 0.1
s = 200  # number of shifts
n0 = 2
p0 = 0

# --- Initialize QMC Sampler ---
sampler = QMC_sampler()
sampler.initialize_from_file('genvec.txt')
for _ in range(9):
    sampler.add_level()

# --- Collect variances per level ---
variances = []
levels = []

# We loop from the highest level down to level 1
for level in range(len(sampler.points) - 1, 0, -1):
    print(f"Processing level {level}")

    fine_pts = sampler.points[level]
    shifts = np.random.rand(s, fine_pts.shape[1])

    # Step 1: Generate paths and evaluate observable
    M1 = pts_to_paths(fine_pts, shifts, T, incr=True)
    P1 = M1.shape[0] // s
    x1 = obj_fctn(euler_maruyama_kuramoto(M1, T, K, sigma, P1), P1)

    # Step 2: Coarsen and split
    M2, M3 = cut_pts(cut_incr(M1))
    P2 = M2.shape[0] // s
    P3 = M3.shape[0] // s

    x2 = obj_fctn(euler_maruyama_kuramoto(M2, T, K, sigma, P2), P2)
    x3 = obj_fctn(euler_maruyama_kuramoto(M3, T, K, sigma, P3), P3)

    # Step 3: Estimate variance of difference
    x_combined = 0.5 * (x2 + x3)
    variance = np.var(x1 - x_combined)
    variances.append(variance)
    levels.append(level)

# --- Plotting ---
levels = np.array(levels)
variances = np.array(variances)

# Reference lines
ref_3 = (2 ** (3 * levels[-2])) * variances[-2] / (2 ** (3 * levels))
ref_4 = (2 ** (4 * levels[-2])) * variances[-2] / (2 ** (4 * levels))

plt.figure(figsize=(6, 4))
plt.semilogy(levels, variances, 'b*-', label="Estimated Variance")
plt.semilogy(levels, ref_3, 'k--', label=r"$2^{-3\ell}$")
plt.semilogy(levels, ref_4, 'k-.', label=r"$2^{-4\ell}$")
plt.xlabel("Level ℓ")
plt.ylabel("Variance of Difference")
plt.title("Multilevel Variance vs. Level")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
