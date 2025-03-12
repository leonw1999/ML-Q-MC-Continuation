import numpy as np
from scipy.stats import norm

def bridge_map(point, t, T):
    X = point.copy()
    stack = [(0, X.shape[1] - 1, t, T)]
    
    while stack:
        left, right, t_left, t_right = stack.pop()
        
        if right - left < 2:
            continue
        
        t_mid = (t_left + t_right) / 2
        mu = X[:, left] + (t_mid - t_left) * (X[:, right] - X[:, left]) / (t_right - t_left)
        sigma = np.sqrt((t_right - t_mid) * (t_mid - t_left) / (t_right - t_left))
        
        mid = left + (right - left) // 2
        X[:, mid] = mu + sigma * norm.ppf(X[:, mid])
        
        stack.append((mid, right, t_mid, t_right))
        stack.append((left, mid, t_left, t_mid))