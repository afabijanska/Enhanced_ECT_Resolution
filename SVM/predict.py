# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:42:55 2019

@author: an_fab
"""

import pickle
import configparser

import numpy as np
import skfuzzy as fuzz
import scipy.io as sio

from models import SVM
from keras.models import model_from_json

from dicts import dict_phantoms

#read config file 

conf = configparser.RawConfigParser()
conf.read('config.txt')

path_test_data = conf.get('data paths','path_test_data')
path_best_weights = conf.get('train settings','best_weights')
path_predicted_phantoms_hard = conf.get('pred settings','path_predicted_phantoms_hard')
path_predicted_phantoms_soft = conf.get('pred settings','path_predicted_phantoms_soft')

#load test data

with open(path_test_data, 'rb') as f:
    X_test = pickle.load(f)
    Y_test = pickle.load(f)
    test_fantom_labels = pickle.load(f)
    test_reconstr_labels = pickle.load(f)
    
X = np.ones((X_test.shape[0],32768,4), dtype = np.float16)
Y = np.ones((X_test.shape[0],32768), dtype = np.float16)    

X[:, 0:X_test.shape[1]] = X_test
X = np.reshape(X, (X.shape[0], X.shape[1], 4))

#load model

model = SVM((X.shape[1], 4))
#model = model_from_json(open('model.json').read())
model.load_weights(path_best_weights)

#predict 

Y_pred = model.predict(X)
Y_pred = Y_pred[:,0:25350]

Y_pred_hard = 2 * np.argmax(Y_pred, axis = -1) + 1
Y_pred_soft = 2 * Y_pred[:,:,1] + 1

#place for predictions

predicted_phantoms_hard = sio.loadmat(path_predicted_phantoms_hard)
phantoms_data = predicted_phantoms_hard['PhantomDataBase']
dataa = np.zeros(phantoms_data.shape)

for i in range(Y_pred.shape[0]):
    
    indx = test_fantom_labels[i]
    dataa[:,indx - 1] = Y_pred_hard[i]

predicted_phantoms_hard['PhantomDataBase'] = dataa
sio.savemat(path_predicted_phantoms_hard,predicted_phantoms_hard)

####

predicted_phantoms_soft = sio.loadmat(path_predicted_phantoms_soft)
phantoms_data = predicted_phantoms_soft['PhantomDataBase']

dataa2 = np.zeros(phantoms_data.shape)

for i in range(Y_pred.shape[0]):
    
    indx =  test_fantom_labels[i]
    dataa2[:,indx - 1] = Y_pred_soft[i]

predicted_phantoms_soft['PhantomDataBase'] = dataa2
sio.savemat(path_predicted_phantoms_soft,predicted_phantoms_soft)

#evaluate

#get phantoms and reconstructions by LPB or 50NL 
path_phantoms = conf.get('data paths','path_phantoms')
path_reconstructions = conf.get('data paths','path_reconstructions')

phantoms = sio.loadmat(path_phantoms)
phantoms_data = phantoms['PhantomDataBase']

reconstructions = sio.loadmat(path_reconstructions)
reconstructions_data = reconstructions[path_reconstructions.replace('.mat','')]


for j in range(0,i+1):
    
    target = phantoms_data[:,test_fantom_labels[j]-1]
    prediction = dataa2[:,test_fantom_labels[j]-1]
    prediction_hard = dataa[:,test_fantom_labels[j]-1]
    
    _input = reconstructions_data[:,test_reconstr_labels[j]-1]
    
    mse = np.sqrt(np.mean((prediction-target)**2))   
    mse1 = np.sqrt(np.mean((prediction_hard-target)**2))
    mse2 = np.sqrt(np.mean((_input-target)**2))   
    
    nmse = fuzz.nmse(target, prediction)
    nmse1 = fuzz.nmse(target, prediction_hard)
    nmse2 = fuzz.nmse(target, _input)
    
    cc = np.corrcoef(prediction, target)
    cc = cc[0,1]*100
    
    cc1 = np.corrcoef(prediction_hard, target)
    cc1 = cc1[0,1]*100
    
    cc2 = np.corrcoef(_input, target) 
    cc2 = cc2[0,1]*100
        
    corr = np.equal(prediction_hard , target).astype(int).sum()
    corr = 100*corr/len(target)
    
    corr2 = np.equal(_input, target).astype(int).sum()
    corr2 = 100*corr/len(target)
    
    print('--------------------------------------------------------------------------')
    print('\t phantom: '+ dict_phantoms[test_fantom_labels[j]] + ' (id: '+ str(test_fantom_labels[j]) + ', reconstr. id: ' + str(test_reconstr_labels[j]) +')')
    print('--------------------------------------------------------------------------')
    print('---- SOFT ----')
    print('mse:\t (pred. soft-target) {:.3f} \t (input-target) {:.3f}'.format(mse, mse2))
    print('nmse:\t (pred. soft-target) {:.3f} \t (input-target) {:.3f}'.format(nmse, nmse2))
    print('cc:\t (pred. soft-target) {:.3f} \t (input-target) {:.3f}'.format(cc, cc2))
    print('---- HARD ----')
    print('mse:\t (pred. hard-target) {:.3f} \t (input-target) {:.3f}'.format(mse1, mse2))
    print('nmse:\t (pred. hard-target) {:.3f} \t (input-target) {:.3f}'.format(nmse1, nmse2))
    print('cc:\t (pred. hard-target) {:.3f} \t (input-target) {:.3f}'.format(cc1, cc2))
    print('acc:\t (pred. hard-target) {:.3f} \t (input-target) {:.3f}'.format(corr, corr2))
    