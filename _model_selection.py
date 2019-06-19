import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.model_selection import KFold

from NN_pr import NN
from NN_pr import NN_pruning_no_sparse as NN_pruning



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

kf=KFold(n_splits=5, random_state=42, shuffle=True)
x=TRAINING[0]
means=[]
end = kf.get_n_splits(x)

for n in [[100],[150],[200],[250],[300]]:
    i=0
    acc=[]
    for train_index, test_index in kf.split(x):
        train_set=[TRAINING[0][train_index], TRAINING[1][train_index]]
        val_set=[TRAINING[0][test_index], TRAINING[1][test_index]]
        
        nn = NN.NN(training=train_set, testing=val_set, lr=0.003, mu=.99, minibatch=100)
        nn.addLayers(n, ['relu'])
        _,acc_val=nn.train(num_epochs=150)
        i+=1
        acc.append(acc_val)
        if i == end:
            m=[n, np.mean(acc)]
            print('Mean accuracy on validation: '+str(m))
            means.append(m)
            
for n in [[100,100],[200,100],[300,100]]:
    i=0
    acc=[]
    for train_index, test_index in kf.split(x):
        train_set=[TRAINING[0][train_index], TRAINING[1][train_index]]
        val_set=[TRAINING[0][test_index], TRAINING[1][test_index]]
                
        nn = NN.NN(training=train_set, testing=val_set, lr=0.003, mu=.99, minibatch=100)
        nn.addLayers(n, ['relu', 'relu'])
        _,acc_val=nn.train(num_epochs=150)
        i+=1
        acc.append(acc_val)
        if i == end:
            m=[n, np.mean(acc)]
            print('Mean accuracy on validation: '+str(m))
            means.append(m)
            
    i=0
    acc1=[]
    for train_index, test_index in kf.split(x):
        train_set=[TRAINING[0][train_index], TRAINING[1][train_index]]
        val_set=[TRAINING[0][test_index], TRAINING[1][test_index]]
                
        nn = NN.NN(training=train_set, testing=val_set, lr=0.003, mu=.99, minibatch=100)
        nn.addLayers(n, ['sigmoid', 'relu'])
        _,acc_val=nn.train(num_epochs=150)
        i+=1
        acc1.append(acc_val)
        if i == end:
            m=[n, np.mean(acc1)]
            print('Mean accuracy on validation: '+str(m))
            means.append(m)


        
        
print(means)
        



