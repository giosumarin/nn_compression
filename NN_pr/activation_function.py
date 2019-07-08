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
        
def linear(x, derivate=False):
    if not derivate:
        return x
    else:
        return np.ones(x.shape)
        
def tanh(z, derivate=False):
    if not derivate:
        ez = np.exp(z)
        enz = np.exp(-z)
        return (ez - enz)/ (ez + enz)
    else:
        return 1 - z**2
        
def LReLU(x, derivate=False):
    alpha=0.01
    if not derivate:
        pos = x * (x > 0)
        neg = x * alpha * (x < 0)
        return pos + neg
    else:
        pos = 1. * (x > 0)
        neg = alpha * (x < 0)
        return pos + neg
