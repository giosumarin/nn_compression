import numpy as np


def ReLU(x, derivate=False):
    if not derivate:
        return x * (x > 0)
    else:
        return 1. * (x > 0)


def sigmoid(x, derivate=False):
    if not derivate:
        return 1 / (1 + np.exp(-x))
    else:
        return x * (1-x)
        
def id(x, derivate=False):
    if not derivate:
        return x
    else:
        return np.ones(x.shape)
