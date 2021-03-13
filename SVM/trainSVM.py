# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:28:31 2019

@author: an_fab
"""

import pickle
import numpy as np
import configparser

from keras.callbacks import ModelCheckpoint,TensorBoard

from models import Unet, SVM
from keras.utils import to_categorical

# read config file

conf = configparser.RawConfigParser()
conf.read('config.txt')
path_train_data = conf.get('data paths','path_train_data')
path_best_weights = conf.get('train settings', 'best_weights')
path_last_weights = conf.get('train settings', 'last_weights')

#training settings

N_epochs = int(conf.get('train settings','num_epochs'))
batch_size = int(conf.get('train settings','batch_size'))

#load train data
 
with open(path_train_data, 'rb') as f:
    X_train = pickle.load(f)
    Y_train = pickle.load(f)
    labels_phantom_train = pickle.load(f)
    labels_reconstr_train = pickle.load(f)
    
X = np.ones((X_train.shape[0],32768,4), dtype = np.float16)
Y = np.ones((Y_train.shape[0],32768), dtype = np.float16)    

#normalize data
    
# X_train = (X_train-1)/2.0
# Y_train = (Y_train-1)/2.0

X[:, 0:X_train.shape[1]] = X_train
Y[:, 0:Y_train.shape[1]] = Y_train
Y = to_categorical(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 4))

#train
model = SVM((X.shape[1], 4))
model._get_distribution_strategy = lambda: None
json_string = model.to_json()
open('model.json', 'w').write(json_string)
checkpointer = ModelCheckpoint(path_best_weights, verbose=1, monitor='val_loss', mode='auto', save_best_only=True) 
tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True, profile_batch = 100000000)
model.fit(X, Y, epochs = N_epochs, batch_size = batch_size, verbose=2, shuffle=True, validation_split=0.2, callbacks=[checkpointer])
model.save_weights(path_last_weights, overwrite=True)