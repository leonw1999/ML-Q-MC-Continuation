#!/usr/bin/env python3
import warnings
import numpy as np
import time
import mimclib
import argparse

from QMC_sampler import QMC_sampler
from pts_to_paths import pts_to_paths
from cut_pts import cut_pts
from cut_incr import cut_incr
from richardson_euler_maruyama_kuramoto import richardson_euler_maruyama_kuramoto
from euler_maruyama_kuramoto import euler_maruyama_kuramoto
from obj_fctn import obj_fctn
import sys

warnings.filterwarnings("error")

def addExtraArguments(parser):
    parser.add_argument("-mlmc_rich", default=False,
                        action="store_true",
                        help="Use Richardson extrapolating")
    parser.add_argument("-mlmc_no_antithetic", default=False,
                        action="store_true",
                        help="Use Richardson extrapolating")
    parser.add_argument("-sde_T", default=1., action="store",
                        help="Number of shifts")
    parser.add_argument("-sde_K", default=1.5, action="store",
                        help="Coupling strength in the Kuramoto system.")
    parser.add_argument("-sde_sig", default=0.1, action="store",
                        help="Noise intensity of the stochastic perturbation.")

def mySampleQoI(run, inds, M):
    sampler = run.sampler
    T, K, sigma = run.params.sde_T, run.params.sde_K, run.params.sde_sig
    solves = np.empty((M, len(inds)), dtype=float)

    tStart = time.process_time()
    assert(len(inds) <= 2)  # Only support two discretizations
    lvl = np.max(inds)
    coarse = len(inds) > 1

    # TODO: Check if adding missing levels invalidates previous samples
    while lvl >= len(sampler.points):
        run.sampler.add_level()

    fine_pts = sampler.points[lvl]
    shifts = run.shift_sampler.random(size=(M, fine_pts.shape[1]))

    # Step 1: Generate paths and evaluate observable
    M1 = pts_to_paths(fine_pts, shifts, T, incr=True)
    # M
    # M1 -> (M*P, N+2)
    P1 = M1.shape[0] // M
    if run.params.mlmc_rich:
        x1 = richardson_euler_maruyama_kuramoto(M1, T, K, sigma, P1)
    else:
        x1 = obj_fctn(euler_maruyama_kuramoto(M1, T, K, sigma, P1), P1)

    solves[:, 0] = x1

    if coarse:
        # Step 2: Coarsen and split
        M2, M3 = cut_pts(cut_incr(M1))
        P2 = M2.shape[0] // M
        P3 = M3.shape[0] // M

        if run.params.mlmc_rich:
            x2 = richardson_euler_maruyama_kuramoto(M2, T, K, sigma, P2)
            x3 = x2
            if not run.params.mlmc_no_antithetic:
                x3 = richardson_euler_maruyama_kuramoto(M3, T, K, sigma, P3)
        else:
            x2 = obj_fctn(euler_maruyama_kuramoto(M2, T, K, sigma, P2), P2)
            x3 = x2
            if not run.params.mlmc_no_antithetic:
                x3 = obj_fctn(euler_maruyama_kuramoto(M3, T, K, sigma, P3), P3)

        solves[:, 1] = 0.5 * (x2 + x3)
    return solves, time.process_time()-tStart

def initRun(run):
    run.sampler = QMC_sampler(p0=0, n0=2)
    run.sampler.initialize_from_file('genvec.txt')
    run.shift_sampler = np.random.default_rng(
        run.params.qoi_seed if hasattr(run.params, "qoi_seed") else None)

    # Default parameters
    run.params.min_dim = 1
    run.params.M0 = np.array([100], dtype=int)
    run.params.s =  getattr(run.params, "s",
                            np.array([4]) if run.params.mlmc_rich
                            else
                            np.array([3]))
    run.params.w =  getattr(run.params, "w", run.params.s/2)
    run.params.gamma =  getattr(run.params, "gamma", np.array([3]))
    run.params.beta =  getattr(run.params, "beta", np.array([2]))
    run.params.max_TOL = getattr(run.params, "max_TOL", 0.5)
    run.params.TOL = getattr(run.params, "TOL", 0.001)

    for _ in range(10):
        run.sampler.add_level()


def run():
    import mimclib.test
    import sys
    mimclib.test.RunStandardTest(fnSampleLvl=mySampleQoI,
                                 fnAddExtraArgs=addExtraArguments,
                                 fnInit=initRun, fnSeed=None)

def plot():
    #!python
    import matplotlib
    matplotlib.use('Agg')

    from mimclib.plot import run_plot_program
    run_plot_program()

def gen_runs(runs=100):
    # Common
    import shlex
    args = " ".join(shlex.quote(arg) for arg in sys.argv[2:])
    cmd = "{} run -db_engine sqlite -db_name qmc.sqlite {}".format(sys.argv[0], args)
    for i in range(runs):
        seed = np.random.randint(2**32-1)
        print("{} -qoi_seed {} -db_tag 'qmc' &".format(cmd, seed))
        seed = np.random.randint(2**32-1)
        print("{} -qoi_seed {} -db_tag 'qmc-rich' -mlmc_rich &".format(cmd, seed))

if __name__ == "__main__":
    cmd = sys.argv[1].lower()
    if cmd == "plot":
        plot()
    elif cmd == "gen":
        gen_runs()
    else:
        run()
