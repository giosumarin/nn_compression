import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys

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

tr = []
te=[]
s=[]

normal_upd = NN.NN.update_layers
for n in [[200,100]]:
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['relu', 'relu'])
    nn.train(num_epochs=150)
    w = (nn.getWeigth())
    for c in [[150,100,50],[200,150,100],[250,200,150]]:
        print("cluster="+str(c))
        w1=np.copy(w)
        nn.layers_shape, nn.centers, nn.idx_layers, nn.v, nn.epoch, nn.cluster = ws.set_ws(c, w1)
        NN.NN.update_layers = ws.ws_update_layers
        NN.NN.updateMomentum = ws.ws_updateMomentum
        nn.train(num_epochs=50)

'''
print("train: "+str(tr))
print("test: "+str(te))
print("space: "+str(s))

fig = plt.figure(figsize=(16,10))
ax=fig.add_(1,1,1)
ax.plot([0,1],tr[0:1],'.-')

for i in range(3):
    ax.plot([0,10,20,30],tr[0+i*4,(i+1)*4],'.-')

ax.legend("50 n")
plt.grid()
plt.plot()


fig.plt.figure(figsize=(16,10))
ax=fig.add_sublot(1,1,1)
for i in range(3):
    ax.plot([0,10,20,30],te[0+i*4,(i+1)*4],'.-')
plt.grid()
plt.plot()

'''
