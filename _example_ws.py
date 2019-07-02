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
print("------------------------no dropout----------------------") 
normal_upd = NN.NN.update_layers
normal_mom = NN.NN.updateMomentum
for n in [[300,100]]: #28*28*300=235200  300*100=3000 100*10=1000
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['relu', 'relu'])
    #nn.train(stop_function=1)#, num_epochs=2)
    nn.train(stop_function=0, num_epochs=120)
    w = (nn.getWeigth())
    for c in [[2352, 30, 10], [1000,150,100],150,[500,200,150],300]:
        print("cluster="+str(c))
        w1=np.copy(w)
        nn.layers_shape, nn.centers, nn.idx_layers, nn.v, nn.epoch, nn.cluster = ws.set_ws(c, w1)
        NN.NN.update_layers = ws.ws_update_layers
        NN.NN.updateMomentum = ws.ws_updateMomentum
        #nn.train(stop_function=1)
        nn.train(stop_function=0, num_epochs=60)
print("------------------------dropout=0.75----------------------")        
for n in [[300,100]]: #28*28*300=235200  300*100=3000 100*10=1000
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100, dropout=0.75)
    NN.NN.update_layers = normal_upd
    NN.NN.updateMomentum = normal_mom
    nn.addLayers(n, ['relu', 'relu'])
    #nn.train(stop_function=1)
    nn.train(stop_function=0, num_epochs=120)
    w = (nn.getWeigth())
    for c in [[2352, 30, 10], [1000,150,100],150,[500,200,150],300]:
        print("cluster="+str(c))
        w1=np.copy(w)
        nn.layers_shape, nn.centers, nn.idx_layers, nn.v, nn.epoch, nn.cluster = ws.set_ws(c, w1)
        NN.NN.update_layers = ws.ws_update_layers
        NN.NN.updateMomentum = ws.ws_updateMomentum
        #nn.train(stop_function=1)
        nn.train(stop_function=0, num_epochs=60)
        
print("------------------------dropout=0.5----------------------")      
for n in [[300,100]]: #28*28*300=235200  300*100=3000 100*10=1000
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100, dropout=0.5)
    NN.NN.update_layers = normal_upd
    NN.NN.updateMomentum = normal_mom
    nn.addLayers(n, ['relu', 'relu'])
    #nn.train(stop_function=1)
    nn.train(stop_function=0, num_epochs=120)
    w = (nn.getWeigth())
    for c in [[2352, 30, 10], [1000,150,100],150,[500,200,150],300]:
        print("cluster="+str(c))
        w1=np.copy(w)
        nn.layers_shape, nn.centers, nn.idx_layers, nn.v, nn.epoch, nn.cluster = ws.set_ws(c, w1)
        NN.NN.update_layers = ws.ws_update_layers
        NN.NN.updateMomentum = ws.ws_updateMomentum
        nn.train(stop_function=0, num_epochs=60)
        #nn.train(stop_function=1)

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
