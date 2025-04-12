import numpy as np

def obj_fctn(x, P):
    M = x.size
    return np.mean(np.exp(- x ** 2).reshape((M // P, P)), axis=1)