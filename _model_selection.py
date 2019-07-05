import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from NN_pr import NN




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

kf=KFold(n_splits=3, random_state=42, shuffle=True)
x=TRAINING[0]
means=[]
end = kf.get_n_splits(x)

#for n in [[50],[100],[150],[200],[250],[300],[20,10],[50,20],[50,50],[80,50],[100,100],[120,80],[150,100],[200,100],[300,100]]:





def mod_sel():    
    n1 = 20    
    n2 = 40
    while n2 <= 300:
        while n1 <= 300:
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
                    print(np.mean(acc))
                    m=[str([n1,n2]), np.mean(acc)]
                    print('Mean accuracy on validation: '+str(m))
                    means.append(m)
        
            n1 += 20
        n2 += 20
        n1 = 20
        
    return means
    
    

'''     



    n1 = 20    
    n2 = 40
    while n2 <= 300:
        while n1 <= 300:
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
                    print(np.mean(acc))
                    m=[str([n1,n2]), np.mean(acc)]
                    print('Mean accuracy on validation: '+str(m))
                    means.append(m)
        
            n1 += 20
        n2 += 20
        n1 = 20       
for n in []:
    i=0
    acc=[]
    for train_index, test_index in kf.split(x):
        train_set=[TRAINING[0][train_index], TRAINING[1][train_index]]
        val_set=[TRAINING[0][test_index], TRAINING[1][test_index]]
                
        nn = NN.NN(training=train_set, testing=val_set, lr=0.003, mu=.99, minibatch=100)
        nn.addLayers(n, ['relu', 'relu'])
        _,acc_val=nn.train(num_epochs=120)
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
    '''

        
        
print(means)
        
'''
[[20, 95.05333333333333], [40, 96.60333333333334], [60, 97.17166666666667], [80, 97.38], [100, 97.52666666666669], [120, 97.64833333333333], [140, 97.70333333333333], [160, 97.745], [180, 97.79666666666667], [200, 97.83166666666666], [220, 97.87833333333333], [240, 97.86166666666668], [260, 97.91166666666668], [280, 97.91666666666667], [300, 97.93499999999999], [320, 95.35333333333331], [320, 96.52], [320, 97.04], [320, 97.17666666666668], [320, 97.33999999999999], [320, 97.47333333333334], [320, 97.41833333333334], [320, 97.545], [320, 97.56], [320, 97.72333333333331], [320, 97.65666666666668], [320, 97.62833333333333], [320, 97.64166666666667], [320, 97.70666666666666], [320, 97.67333333333333]]
20,20 40,10 ....


'''


