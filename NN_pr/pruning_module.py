import numpy as np
import math
from scipy.sparse import csc_matrix, issparse
from NN_pr import NN 
from NN_pr import logger as log
from NN_pr import activation_function as af

def set_pruned_layers(pruning, weights):
    layers = weights
    num_layers = len(layers)
    mask = []
    v = []
    epoch = 0
	    
    for i in range(num_layers):
        W=layers[i][0]
        m = np.abs(W) > np.percentile(np.abs(W), pruning)
        mask.append(m)	  
        W_pruned = W * m
        layers[i][0] = W_pruned
        v.append([0, 0])
    return layers, mask, v, epoch

def mask_update_layers(self, deltasUpd, momentumUpdate):
    for i in range(self.nHidden + 1):
        self.layers[i][0] += (deltasUpd[i][0] + momentumUpdate * self.v[i][0]) * self.mask[i]
        self.layers[i][1] += deltasUpd[i][1] + momentumUpdate * self.v[i][1]
    self.v = deltasUpd
