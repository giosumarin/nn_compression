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
'''
for n in [[250,100]]:
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['relu', 'relu','tanh'])
    a,b=nn.train(stop_function=0, num_epochs=150)
    w = (nn.getWeigth())
    for p in [10,20]:
        print("Pruning="+str(p)+"%")
        w1=np.copy(w)
        pr.set_pruned_layers(nn, p, w1)
        nn.train(stop_function=0, num_epochs=50)'''
        
for n in [[250,100]]:
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['leakyrelu', 'leakyrelu','sigmoid'])
    a,b=nn.train(stop_function=0, num_epochs=150)
    w = (nn.getWeigth())
    for p in [10,20]:
        print("Pruning="+str(p)+"%")
        w1=np.copy(w)
        pr.set_pruned_layers(nn, p, w1)
        nn.train(stop_function=0, num_epochs=50)
        
for n in [[250,100]]:
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['leakyrelu', 'leakyrelu','tanh'])
    a,b=nn.train(stop_function=0, num_epochs=150)
    w = (nn.getWeigth())
    for p in [10,20]:
        print("Pruning="+str(p)+"%")
        w1=np.copy(w)
        pr.set_pruned_layers(nn, p, w1)
        nn.train(stop_function=0, num_epochs=50)
        
print("#"*30)    
    
for n in [[250,100]]:
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100, dropout=0.75)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['relu', 'relu','tanh'])
    a,b=nn.train(stop_function=0, num_epochs=150)
    w = (nn.getWeigth())
    for p in [10,20]:
        print("Pruning="+str(p)+"%")
        w1=np.copy(w)
        pr.set_pruned_layers(nn, p, w1)
        nn.train(stop_function=0, num_epochs=50)
        
for n in [[250,100]]:
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100, dropout=0.75)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['leakyrelu', 'leakyrelu','sigmoid'])
    a,b=nn.train(stop_function=0, num_epochs=150)
    w = (nn.getWeigth())
    for p in [10,20]:
        print("Pruning="+str(p)+"%")
        w1=np.copy(w)
        pr.set_pruned_layers(nn, p, w1)
        nn.train(stop_function=0, num_epochs=50)
        
for n in [[250,100]]:
    nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100, dropout=0.75)
    NN.NN.update_layers = normal_upd
    nn.addLayers(n, ['leakyrelu', 'leakyrelu','tanh'])
    a,b=nn.train(stop_function=0, num_epochs=150)
    w = (nn.getWeigth())
    for p in [10,20]:
        print("Pruning="+str(p)+"%")
        w1=np.copy(w)
        pr.set_pruned_layers(nn, p, w1)
        nn.train(stop_function=0, num_epochs=50)







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
