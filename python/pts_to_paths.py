import numpy as np
from scipy.stats import norm
from bridge_map import bridge_map
import matplotlib.pyplot as plt


def pts_to_paths(pts, shifts, T, incr=True):

    m, n = shifts.shape
    k, _ = pts.shape

    # Repeat A for each row in B
    shifts_expanded = np.repeat(shifts, k, axis=0)

    # Tile B to match the repeated A
    pts_tiled = np.tile(points, (m, 1))

    p_measure_pts = A_expanded + B_tiled % 1

    p_measure_pts = np.zeros(qmc_pts.shape)

    # init
    p_measure_pts[:, 0] = norm.ppf(qmc_pts[:, 0])

    # vp
    p_measure_pts[:, 1] = (qmc_pts[:, 1] - 0.5) * (2 / 5)

    # Gaussian
    v = np.hstack([np.zeros((qmc_pts.shape[0], 1)), qmc_pts[:, 2:]])
    v[:, -1] = np.sqrt(T) * norm.ppf(qmc_pts[:, -1])


    v = bridge_map(v, 0, T)

    if incr:
        p_measure_pts[:, 2:] = v[:, 1:] - v[:, :(-1)]
    else:
        p_measure_pts[:, 2:] = v[:, 1:]


    return p_measure_pts