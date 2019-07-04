import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys
from joblib import Parallel, delayed, load
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans

from NN_pr import NN
from NN_pr import WS_module as ws


RANDOM_SEED = 42
N_CLASSES = 10

TRAINING=load("data/trainjoblib", mmap_mode='r')
TESTING=load("data/testjoblib", mmap_mode='r')


sil=[]
normal_upd = NN.NN.update_layers
normal_mom = NN.NN.updateMomentum
#28*28*300=235200  300*100=3000 100*10=1000
nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100, dropout=0.75)
NN.NN.update_layers = normal_upd
nn.addLayers([25,10], ['relu', 'relu'])
#nn.train(stop_function=1)#, num_epochs=2)
nn.train(stop_function=0, num_epochs=1)
w1 = nn.getWeigth()

#with open('pesi', 'wb') as f:
#    pickle.dump(w1, f)

best=[[0,0],[0,0],[0,0]]

def silhouette(i):
    w=np.copy(w1)
    ncl=2**i
    silinner=[]
    s=[]
    for j in range(len(w)):
        if (w[j][0].shape[0]*w[j][0].shape[1] >= ncl):
            cl = MiniBatchKMeans(n_clusters = ncl, random_state = 42, init_size = 3 * ncl)
            X=np.hstack(w[j][0]).reshape(-1,1)
            cl_lb = cl.fit_predict(X)
            index=round(silhouette_score(X, cl_lb)*100, 3)
            if index > best[j][0]:
                best[j][0] = index
                best[j][1] = ncl
            silinner.append(index)
        else:
            silinner.append("Nope")    
    sil.append([ncl, silinner])
    print(sil) 
    return sil
    
print(Parallel(n_jobs=-1)(delayed(silhouette)(k) for k in range(6,8,1)))
print(best)

'''
nn = NN.NN(training=TRAINING, testing=TESTING, lr=0.003, mu=.99, minibatch=100)
nn.addLayers([50,50], ['relu', 'relu'])
#nn.train(stop_function=1)
nn.train(stop_function=0, num_epochs=10)
w = (nn.getWeigth())
w1=np.copy(w)
ws.set_ws(nn, 30, w1)
nn.train(stop_function=0, num_epochs=10)

  
    for c in [100, 200, [100, 20, 10],[1176, 15, 5], [2352, 30, 10]]:
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
    for c in [[2352, 30, 10]]:#, [1000,150,100],150,[500,200,150],300]:
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
    for c in [[2352, 30, 10]]:#, [1000,150,100],150,[500,200,150],300]:
        print("cluster="+str(c))
        w1=np.copy(w)
        nn.layers_shape, nn.centers, nn.idx_layers, nn.v, nn.epoch, nn.cluster = ws.set_ws(c, w1)
        NN.NN.update_layers = ws.ws_update_layers
        NN.NN.updateMomentum = ws.ws_updateMomentum
        nn.train(stop_function=0, num_epochs=60)
        #nn.train(stop_function=1)
'''
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



    #with open('pesi', 'rw') as f:
    #    pickle.dump(w, f)
        
    #with open('pesi', 'rb') as f:
    #    wOld=pickle.load(f)
