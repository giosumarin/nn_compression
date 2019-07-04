import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys
from joblib import Parallel, delayed, load
from sklearn.model_selection import KFold
from itertools import product

from NN_pr import NN




DATA_PATH = 'data/mnist/'

IMAGES_TRAIN = 'data_training'
IMAGES_TEST = 'data_testing'

RANDOM_SEED = 42
N_CLASSES = 10

TRAINING=load("data/trainjoblib", mmap_mode='r')
TESTING=load("data/testjoblib", mmap_mode='r')

def mod_sel(n1,n2):    
    kf=KFold(n_splits=3, random_state=42, shuffle=True)
    x=TRAINING[0]
    means=[]
    end = kf.get_n_splits(x)
    i=0
    acc=[]
    for train_index, test_index in kf.split(x):
        train_set=[TRAINING[0][train_index], TRAINING[1][train_index]]
        val_set=[TRAINING[0][test_index], TRAINING[1][test_index]]
        
        nn = NN.NN(training=train_set, testing=val_set, lr=0.003, mu=.99, minibatch=100)
        nn.addLayers([n1,n2], ['relu','relu'])
        _,acc_val=nn.train(stop_function=0,num_epochs=120)
        i+=1
        acc.append(acc_val)
        if i == end:
            m=[str([n1,n2]), round(np.mean(acc),2)]
            print('Mean accuracy on validation: '+str(m))
            means.append(m)
    return means
    
    


print(Parallel(n_jobs=-1)(delayed(mod_sel)(n1,n2) for n1, n2 in product(range(200,301,25), range(60,121,20))))
#8m20s

#for n1, n2 in product(range(10,20,5), range(10,20,5)):
#    print(mod_sel(n1,n2)) 
#9m15s
