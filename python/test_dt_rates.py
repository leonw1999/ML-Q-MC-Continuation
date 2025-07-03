import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from QMC_sampler import QMC_sampler
from pts_to_paths import pts_to_paths
from cut_incr import cut_incr
from euler_maruyama_kuramoto import euler_maruyama_kuramoto
from obj_fctn import obj_fctn

plt.rcParams.update({'font.size': 11})
plt.rcParams['lines.linewidth'] = 1.05

# --- Initialize QMC Sampler ---
sampler = QMC_sampler()
sampler.initialize_from_file('genvec.txt')
for _ in range(7):
    sampler.add_level()

# --- Parameters ---
T = 1.0
K = 1.5
sigma = 0.1
s = 200  # number of shifts
initial_pts = sampler.points[-1]  # Use the finest level
shifts = np.random.rand(s, initial_pts.shape[1])

# Step 1: Transform points to paths
current_pts = pts_to_paths(initial_pts, shifts, T, incr=True)

# --- Tracking results ---
N_values = []
variances = []
means = []
mean_diffs = []

prev_means = None

# --- Recursive cutting of time steps ---
while (current_pts.shape[1] - 2) >= 2:
    N = current_pts.shape[1] - 2
    P = int(current_pts.shape[0] / s)
    N_values.append(N)

    # Step 2: Simulate with Euler-Maruyama
    results = euler_maruyama_kuramoto(current_pts, T, K, sigma, P)

    # Step 3: Evaluate objective function
    outputs = obj_fctn(results, P)

    # Step 4: Collect statistics
    means.append(np.mean(outputs))
    variances.append(np.var(outputs))

    if prev_means is not None:
        mean_diffs.append(np.abs(means[-1] - prev_means))
    else:
        mean_diffs.append(0.0)

    prev_means = means[-1]

    print(f"N = {N}, P = {P}, shape = {current_pts.shape}")

    # Step 5: Cut matrix in time steps
    current_pts = cut_incr(current_pts)

# --- Plotting ---
N_values = np.array(N_values)

# N^{-2} reference line for comparison
ref_line1 = (1 / N_values) * variances[-1] * N_values[-1]
ref_line2 = (1 / N_values[1:]) * mean_diffs[-1] * N_values[-1]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(N_values, variances, 'r*-', label="Variance")
plt.loglog(N_values, ref_line1, 'k--', dashes=(8, 4), label="$N^{-1}$")
plt.xlabel("N (number of time steps)")
plt.ylabel("Sample Variance")
plt.title("Variance of Objective vs. Time Resolution")
plt.legend()

plt.subplot(1, 2, 2)
plt.loglog(N_values[1:], mean_diffs[1:], 'b*-', label="Mean Differences")
plt.loglog(N_values[1:], ref_line2, 'k--', dashes=(8, 4), label="$N^{-1}$")
plt.xlabel("N (number of time steps)")
plt.ylabel("Change in Mean")
plt.title("Mean Convergence vs. Time Resolution (Const expected)")
plt.legend()

plt.tight_layout()
plt.show()
