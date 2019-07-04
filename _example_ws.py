import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans

from NN_pr import NN
from NN_pr import WS_module as ws



DATA_PATH = 'data/mnist/'

IMAGES_TRAIN = 'data_training'
IMAGES_TEST = 'data_testing'

RANDOM_SEED = 42
N_CLASSES = 10


data_training = DATA_PATH+IMAGES_TRAIN
data_testing = DATA_PATH+IMAGES_TEST
ft = gzip.open(data_training, 'rb')
TRAINING = pickle.load(ft)
ft.close()
ft = gzip.open(data_testing, 'rb')
TESTING = pickle.load(ft)
ft.close()

    #with open('pesi', 'rw') as f:
    #    pickle.dump(w, f)
        
    #with open('pesi', 'rb') as f:
    #    wOld=pickle.load(f)

nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100)
nn.addLayers([50,50], ['relu', 'relu'])
#nn.train(stop_function=1)
nn.train(stop_function=0, num_epochs=10)
w = (nn.getWeigth())
w1=np.copy(w)
ws.set_ws(nn, 30, w1)
nn.train(stop_function=0, num_epochs=10)

