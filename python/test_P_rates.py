import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from QMC_sampler import QMC_sampler
from pts_to_paths import pts_to_paths
from cut_pts import cut_pts
from euler_maruyama_kuramoto import euler_maruyama_kuramoto
from obj_fctn import obj_fctn

plt.rcParams.update({'font.size': 11})
plt.rcParams['lines.linewidth'] = 1.05


# --- Initialize QMC Sampler ---
sampler = QMC_sampler()
sampler.initialize_from_file('/Users/wilkoslp/Desktop/Repositories/ML-Q-MC-Continuation/python/genvec.txt')
for _ in range(7):
    sampler.add_level()

# --- Parameters ---
T = 1.0
K = 1.5
sigma = 0.1
s = 20000 # number of shifts
initial_pts = sampler.points[-1]  # Use the finest level
shifts = np.random.rand(s, initial_pts.shape[1])
# Step 1: Transform points to paths
current_pts = pts_to_paths(initial_pts, shifts, T, incr=True)

# --- Tracking results ---
P_values = []
variances = []
means = []
mean_diffs = []

prev_means = None

# --- Recursive cutting and simulation ---
while int(current_pts.shape[0]/s) >= 2:
    P = int(current_pts.shape[0]/s)
    P_values.append(P)

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

    print(f"P = {P}, shape = {current_pts.shape}")

    # Step 5: Cut matrix for next iteration
    current_pts, _ = cut_pts(current_pts)

# --- Plotting ---
P_values = np.array(P_values)

# P^{-2} reference line for comparison
ref_line1 = (1 / P_values**2) * variances[-1] * P_values[-1]**2  # scaled to match first variance
ref_line2 = (1 / P_values[1:]**2) * mean_diffs[-1] * P_values[-1]**2  # scaled to match first variance

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.loglog(P_values, variances,'r*-', label="Variance")
plt.loglog(P_values, ref_line1, 'k--', dashes=(8, 4), label="$P^{-2}$")
plt.xlabel("P (number of samples)")
plt.ylabel("Sample Variance")
plt.title("Variance of Objective vs. Sample Count")
plt.legend()

plt.subplot(1, 2, 2)
plt.loglog(P_values[1:], mean_diffs[1:], 'b*-', label="Mean Differences")
plt.loglog(P_values[1:], ref_line2, 'k--', dashes=(8, 4), label="$P^{-2}$")
plt.xlabel("P (number of samples)")
plt.ylabel("Change in Mean")
plt.title("Mean Convergence vs. Sample Count")
plt.legend()

plt.tight_layout()
plt.show()
