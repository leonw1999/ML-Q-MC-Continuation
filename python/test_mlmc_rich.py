import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from QMC_sampler import QMC_sampler
from pts_to_paths import pts_to_paths
from cut_pts import cut_pts
from cut_incr import cut_incr
from richardson_euler_maruyama_kuramoto import richardson_euler_maruyama_kuramoto
from obj_fctn import obj_fctn

[plt].rcParams.update({'font.size': 11})
plt.rcParams['lines.linewidth'] = 1.05

# --- Parameters ---
max_level = 9
T = 1.0
K = 1.5
sigma = 0.1
s = 100  # number of shifts
p0 = 0
n0 = 2

# --- Initialize QMC Sampler ---
sampler = QMC_sampler(p0=p0, n0=n0)
sampler.initialize_from_file('genvec.txt')
for _ in range(max_level):
    sampler.add_level()

# --- Collect variances per level ---
variances = []
levels = []

# We loop from the highest level down to level 1
for level in range(len(sampler.points) - 1, 1, -1):
    print(f"Processing level {level}")

    fine_pts = sampler.points[level]
    shifts = np.random.rand(s, fine_pts.shape[1])

    # Step 1: Generate paths and evaluate observable
    M1 = pts_to_paths(fine_pts, shifts, T, incr=True)
    P1 = M1.shape[0] // s
    x1 = richardson_euler_maruyama_kuramoto(M1, T, K, sigma, P1)

    # Step 2: Coarsen and split
    M2, M3 = cut_pts(cut_incr(M1))
    P2 = M2.shape[0] // s
    P3 = M3.shape[0] // s

    print(P2, P3)

    '''
    block_size = P2 + P3
    num_blocks = cut_incr(M1).shape[0] // block_size

    # Get indices for M2 (first P2 rows in each block)
    idx_M2 = np.concatenate([
        np.arange(i * block_size, i * block_size + P2)
        for i in range(num_blocks)
    ])

    # Get indices for M3 (next P3 rows in each block)
    idx_M3 = np.concatenate([
        np.arange(i * block_size + P2, (i + 1) * block_size)
        for i in range(num_blocks)
    ])

    M2 = cut_incr(M1)[idx_M2]
    M3 = cut_incr(M1)[idx_M3]
    '''


    x2 = richardson_euler_maruyama_kuramoto(M2, T, K, sigma, P2)
    x3 = richardson_euler_maruyama_kuramoto(M3, T, K, sigma, P3)

    # Step 3: Estimate variance of difference
    x_combined = 0.5 * (x2 + x3)
    variance = np.var(x1 - x_combined)
    variances.append(variance)
    levels.append(level)

# --- Plotting ---
levels = np.array(levels)
variances = np.array(variances)

# Reference lines
ref_3 = (2 ** (3 * levels[-6])) * variances[-6] / (2 ** (3 * levels))
ref_4 = (2 ** (4 * levels[-6])) * variances[-6] / (2 ** (4 * levels))

plt.figure(figsize=(6, 4))
plt.semilogy(levels, variances, 'b*-', label="Estimated Variance")
#plt.semilogy(levels, ref_3, 'k--', label=r"$2^{-3\ell}$")
plt.semilogy(levels, ref_4, 'k-.', label=r"$2^{-4\ell}$")
plt.xlabel("Level ℓ")
plt.ylabel("Variance of Difference, Richardson")
plt.title("Multilevel Variance vs. Level")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
