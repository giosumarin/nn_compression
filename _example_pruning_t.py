import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys

from NN_pr import NN
from NN_pr import pruning_module as pr



DATA_PATH = 'data/mnist/'

IMAGES_TRAIN = 'data_training'
IMAGES_TEST = 'data_testing'

RANDOM_SEED = 42
N_CLASSES = 10

with open('dataset_tools', 'rb') as file:
    tools = pickle.load(file)
    
X_scaled = tools['X_scaled']
y_scaled = tools['y_scaled']
X_test_scaled = tools['X_test_scaled']
y_test_scaled = tools['y_test_scaled']

tr = []
te=[]
s=[]
normal_upd = NN.NN.update_layers
for n in [[20],[50]]:
    nn = NN.NN(training=[X_scaled, y_scaled], testing=[X_test_scaled, y_test_scaled], lr=0.003, mu=.99, minibatch=100)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['relu'])
    nn.set_output_id_fun()
    a,b=nn.train(num_epochs=10)
    w = (nn.getWeigth())
    for p in [10,20,30,40]:
        print("Pruning="+str(p)+"%")
        w1=np.copy(w)
        nn.layers, nn.mask, nn.v, nn.epoch = pr.set_pruned_layers(p, w1)
        NN.NN.update_layers = pr.mask_update_layers
        a,b=nn.train(num_epochs=10)


