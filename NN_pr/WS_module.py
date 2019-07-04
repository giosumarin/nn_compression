import numpy as np
import math
from sklearn.cluster import KMeans,MiniBatchKMeans
import scipy.ndimage
from NN_pr import NN 
from NN_pr import logger as log
from NN_pr import activation_function as af

def nearest_centroid_index(centers,value):
    centers = np.asarray(centers)
    idx = (np.abs(centers - value)).argmin()
    return idx

def build_clusters(cluster,weights):
    kmeans = MiniBatchKMeans(n_clusters=cluster,init_size=3*cluster)
    kmeans.fit(np.hstack(weights).reshape(-1,1))
    return kmeans.cluster_centers_

def redefine_weights(weights,centers):
    arr_ret = np.empty_like(weights).astype(np.int16)
    for i, row in enumerate(weights):
        for j, col in enumerate(row):
            arr_ret[i,j] = nearest_centroid_index(centers,weights[i,j])
    return arr_ret

def idx_matrix_to_matrix(idx_matrix,centers,shape):
    return centers[idx_matrix.reshape(-1,1)].reshape(shape)

def centroid_gradient_matrix(idx_matrix,gradient,cluster):
    return scipy.ndimage.sum(gradient,idx_matrix,index=range(cluster))
    #provare mean
    
def set_ws(nn, cluster, weights):
    layers_shape = []
    centers = []
    idx_layers = []
    v = []
    
    if isinstance(cluster, int):
        cluster = [cluster]*len(weights)

    for i in range(len(weights)):
        layers_shape.append(weights[i][0].shape)
        centers.append(build_clusters(cluster[i], weights[i][0]))
        idx_layers.append([redefine_weights(weights[i][0],centers[i]), weights[i][1]])  
        v.append([0,0])
    NN.NN.update_layers = ws_update_layers
    NN.NN.updateMomentum = ws_updateMomentum
    
    nn.layers_shape = layers_shape
    nn.centers = centers
    nn.idx_layers = idx_layers
    nn.v = v
    nn.epoch = 0
    nn.cluster = cluster 

    
def ws_update_layers(self, deltasUpd, momentumUpdate):
    v_temp = []
    for i in range(self.nHidden + 1):
        cg = centroid_gradient_matrix(self.idx_layers[i][0],deltasUpd[i][0],self.cluster[i])  
            
        self.centers[i] += np.array(cg).reshape(self.cluster[i],1)

        self.layers[i][1] += deltasUpd[i][1] + momentumUpdate * self.v[i][1]
        v_temp.append([cg, deltasUpd[i][1]])
    self.v=v_temp
        
    
    
def ws_updateMomentum(self, X, t, nEpochs, learningRate, momentumUpdate):
    numBatch = (int)(self.numEx / self.minibatch)
    max_learning_rate = learningRate
    min_learning_rate = 0.0001
    decay_speed = 100.0
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-self.epoch/decay_speed)

    for nb in range(numBatch):
        indexLow = nb * self.minibatch
        indexHigh = (nb + 1) * self.minibatch
        self.layers = []
        
        for i in range(self.nHidden + 1):
            self.layers.append([idx_matrix_to_matrix(self.idx_layers[i][0], self.centers[i], self.layers_shape[i]), self.idx_layers[i][1]])    
                  
        outputs = self.predict(X[indexLow:indexHigh])
        if self.p != None:
            for i in range(len(outputs) - 1):
                mask = (np.random.rand(*outputs[i].shape) < self.p) / self.p
                outputs[i] *= mask

        y = outputs[-1]
        deltas = []
        deltas.append(self.act_fun[-1](y, True) * (y - t[indexLow:indexHigh]))

        for i in range(self.nHidden):
            deltas.append(
                np.dot(deltas[i], self.layers[self.nHidden - i][0].T) * self.act_fun[self.nHidden - i - 1](outputs[self.nHidden - i - 1], True))
        deltas.reverse()

        deltasUpd = []
        deltasUpd.append([- lr * np.dot(X[indexLow:indexHigh].T, deltas[0]), - lr * np.sum(deltas[0], axis=0, keepdims=True)])
        for i in range(self.nHidden):
            deltasUpd.append(
                [- lr * np.dot(outputs[i].T, deltas[i + 1]), - lr * np.sum(deltas[i + 1], axis=0, keepdims=True)])

        self.update_layers(deltasUpd, momentumUpdate)
    

    

