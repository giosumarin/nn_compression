import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import sys
from joblib import Parallel, delayed, dump, load
from sklearn.model_selection import KFold
from itertools import product

from NN_pr import NN




DATA_PATH = 'mnist/'

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
dump(TRAINING, "trainjoblib")
dump(TESTING, "testjoblib")
