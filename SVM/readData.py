# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:34:24 2019

@author: an_fab
"""

import torch
import scipy
import random
import pickle
import numpy as np
import scipy.io as sio

from dicts import dict_phantoms, ECTModels

import configparser
        
#read config file to get some settings

conf = configparser.RawConfigParser()
conf.read('config.txt')

#alg settings
method = conf.get('alg settings','method')                    #reconstruction algorithm
config = conf.get('alg settings','config')                    #configuration of electrodes
num_phantoms = int(conf.get('alg settings','num_phantoms'))   #total num of phantoms
num_test = int(conf.get('alg settings','num_test'))           #number of phantoms used for testing...
num_train = num_phantoms-num_test                               #and for training

#data paths 
path_phantoms = conf.get('data paths','path_phantoms')
path_reconstructions = conf.get('data paths','path_reconstructions')
path_graph_structure = conf.get('data paths','path_graph_structure')
path_train_data = conf.get('data paths','path_train_data')
path_test_data = conf.get('data paths','path_test_data')

#read data
mat = sio.loadmat(path_graph_structure)
csimp = mat['Csimp']
temp = np.absolute(csimp)
csimp = csimp/temp.max()
# simp = mat['simp']
# vtx = mat['vtx']

phantoms = sio.loadmat(path_phantoms)
phantoms_data = phantoms['PhantomDataBase']

reconstructions = sio.loadmat(path_reconstructions)
reconstructions_data = reconstructions[path_reconstructions.replace('.mat','')]

#generate train set

index_list = list(range(num_phantoms))
random.shuffle(index_list)
random.shuffle(index_list)

X_train = []
Y_train = []
labels_phantom_train = []
labels_reconstr_train = []

for i in range(0,num_train):
        
    indx = index_list[i] + 1
    indx2 = (num_phantoms *( ECTModels[config] - 1) ) + indx - 1

    r = reconstructions_data[:,indx2]
    r = (r-1)/2
    e = np.zeros((25350,4))
    e[:,0] = r
    e[:,1:4] = csimp
    X_train.append(e)

    y = phantoms_data[:,indx-1]
    y = (y-1)/2
    Y_train.append(y)    
            
    labels_phantom_train.append(indx)
    labels_reconstr_train.append(indx2+1)

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
labels_phantom_train = np.asarray(labels_phantom_train)
labels_reconstr_train = np.asarray(labels_reconstr_train)

# generate test set

X_test = []
Y_test = []
labels_phantom_test = []
labels_reconstr_test = []

for i in range(num_train, num_phantoms):
      
    indx = index_list[i] + 1
    indx2 = (num_phantoms *( ECTModels[config] - 1) ) + indx -1
    
    phantom = dict_phantoms[indx]
    print(phantom)
  
    r = reconstructions_data[:,indx2]
    r = (r-1)/2
    e = np.zeros((25350,4))
    e[:,0] = r
    e[:,1:4] = csimp
    X_test.append(e)
    
    y = phantoms_data[:,indx - 1]
    y = (y-1)/2
    Y_test.append(y)    
            
    labels_phantom_test.append(indx)
    labels_reconstr_test.append(indx2+1)
    
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)
labels_phantom_test = np.asarray(labels_phantom_test)
labels_reconstr_test = np.asarray(labels_reconstr_test)

#save data

with open(path_train_data, 'wb') as f:
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(labels_phantom_train, f)
    pickle.dump(labels_reconstr_train, f)

with open(path_test_data, 'wb') as f:
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)    
    pickle.dump(labels_phantom_test, f)
    pickle.dump(labels_reconstr_test, f)