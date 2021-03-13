# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 00:46:20 2019

@author: an_fab
"""

from keras.regularizers import l2

from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.layers import Input, InputLayer, Conv1D, MaxPooling1D, UpSampling1D, Dropout, concatenate, BatchNormalization, Dense, Activation
from keras.optimizers import Adam


def SVM(input_size):
    
    # inputs = Input(input_size)
    # dens1 = Dense(64, activation = 'relu')(inputs)
    # dens2 = Dense(1, W_regularizer=l2(0.01))(dens1)
    # activ = Activation('linear')(dens2)
    
    # model = Sequential(input = inputs, output = activ)
    # model.compile(loss='hinge', optimizer='adadelta', metrics=['accuracy'])
    
    # model = Sequential()
    # model.add(Input(shape=input_size))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(2, kernel_regularizer=l2(0.01)))
    # model.add(Activation('linear'))
    # model.compile(loss='hinge',
    #           optimizer='adadelta',
    #           metrics=['accuracy'])
    
    model = Sequential(
    [
        InputLayer(input_shape=input_size),
        Dense(64, activation='relu'),
        Dense(2, kernel_regularizer=l2(0.01)),
        Activation('linear'),
    ]
    )
 
    model.compile(loss='hinge',optimizer='adadelta', metrics=['accuracy'])
    model.summary()
    return model
    
def Unet(input_size, pool_size):
    
    inputs = Input(input_size)

    conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same')(bn1)
    bn1 = BatchNormalization()(conv1)
#    conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same')(bn1)
#    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=(pool_size))(bn1)

    conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same')(bn2)
    bn2 = BatchNormalization()(conv2)
#    conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same')(bn2)
#    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=(pool_size))(bn2)

    conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same')(bn3)
    bn3 = BatchNormalization()(conv3)
#    conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same')(bn3)
#    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=(pool_size))(bn3)

    conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same')(bn4)
    bn4 = BatchNormalization()(conv4)
#    conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same')(bn4)
#    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=(pool_size))(bn4)

    conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same')(bn5)
    bn5 = BatchNormalization()(conv5)
#    conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same')(bn5)
#    bn5 = BatchNormalization()(conv5)
    
    #drop5 = Dropout(0.5)(conv5)

    up6 = Conv1D(512, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(bn5))
    merge6 = concatenate([conv4,up6], axis = -1)
    drop6 = Dropout(0.2)(merge6)
    conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same')(drop6)
    conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same')(conv6)
#    conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv1D(256, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(conv6))
    merge7 = concatenate([conv3,up7], axis = -1)
    drop7 = Dropout(0.2)(merge7)
    conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same')(drop7)
    conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same')(conv7)
#    conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv1D(128, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(conv7))
    merge8 = concatenate([conv2,up8], axis = -1)
    drop8 = Dropout(0.2)(merge8)
    conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same')(drop8)
    conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same')(conv8)
#    conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same')(conv8)
    
    up9 = Conv1D(64, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(conv8))
    merge9 = concatenate([conv1,up9], axis = -1)
    drop9 = Dropout(0.2)(merge9)
    conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same')(drop9)
    conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same')(conv9)
#    conv9 = Conv1D(2, 3, activation = 'relu', padding = 'same')(conv9)
    
    conv10 = Conv1D(2, 1, activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    #model.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mae', 'acc'])
    model.compile(optimizer = Adam(lr = 1e-3), loss = 'mean_squared_error', metrics = ['acc'])
    
    model.summary()

    return model

####

def UnetDeep(input_size, pool_size):
    
    inputs = Input(input_size)

    conv1 = Conv1D(8, 3, activation = 'relu', padding = 'same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv1D(8, 3, activation = 'relu', padding = 'same')(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=(pool_size))(bn1)

    conv2 = Conv1D(16, 3, activation = 'relu', padding = 'same')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv1D(16, 3, activation = 'relu', padding = 'same')(bn2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=(pool_size))(bn2)

    conv3 = Conv1D(32, 3, activation = 'relu', padding = 'same')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv1D(32, 3, activation = 'relu', padding = 'same')(bn3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=(pool_size))(bn3)

    conv4 = Conv1D(64, 3, activation = 'relu', padding = 'same')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv1D(64, 3, activation = 'relu', padding = 'same')(bn4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling1D(pool_size=(pool_size))(bn4)

    conv5 = Conv1D(128, 3, activation = 'relu', padding = 'same')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv1D(128, 3, activation = 'relu', padding = 'same')(bn5)
    bn5 = BatchNormalization()(conv5)
    pool5 = MaxPooling1D(pool_size=(pool_size))(bn4)
    
    conv6 = Conv1D(256, 3, activation = 'relu', padding = 'same')(pool5)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv1D(256, 3, activation = 'relu', padding = 'same')(bn6)
    bn6 = BatchNormalization()(conv6)
    pool6 = MaxPooling1D(pool_size=(pool_size))(bn6)
    
    conv7 = Conv1D(512, 3, activation = 'relu', padding = 'same')(pool6)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv1D(512, 3, activation = 'relu', padding = 'same')(bn7)
    bn7 = BatchNormalization()(conv7)
    
    up8 = Conv1D(256, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(bn7))
    merge8 = concatenate([conv6,up8], axis = -1)
    drop8 = Dropout(0.2)(merge8)
    conv8 = Conv1D(256, 3, activation = 'relu', padding = 'same')(drop8)
    conv8 = Conv1D(256, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv1D(128, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(conv8))
    merge9 = concatenate([conv5,up9], axis = -1)
    drop9 = Dropout(0.2)(merge9)
    conv9 = Conv1D(128, 3, activation = 'relu', padding = 'same')(drop9)
    conv9 = Conv1D(128, 3, activation = 'relu', padding = 'same')(conv9)

    up10 = Conv1D(65, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(conv9))
    merge10 = concatenate([conv4,up10], axis = -1)
    drop10 = Dropout(0.2)(merge10)
    conv10 = Conv1D(64, 3, activation = 'relu', padding = 'same')(drop10)
    conv10 = Conv1D(64, 3, activation = 'relu', padding = 'same')(conv10)
    
    up11 = Conv1D(32, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(conv10))
    merge11 = concatenate([conv3,up11], axis = -1)
    drop11 = Dropout(0.2)(merge11)
    conv11 = Conv1D(32, 3, activation = 'relu', padding = 'same')(drop11)
    conv11 = Conv1D(32, 3, activation = 'relu', padding = 'same')(conv11)
 
    up12 = Conv1D(16, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(conv11))
    merge12 = concatenate([conv2,up12], axis = -1)
    drop12 = Dropout(0.2)(merge12)
    conv12 = Conv1D(16, 3, activation = 'relu', padding = 'same')(drop12)
    conv12 = Conv1D(16, 3, activation = 'relu', padding = 'same')(conv12)

    up13 = Conv1D(8, 2, activation = 'relu', padding = 'same')(UpSampling1D(size = (pool_size))(conv11))
    merge13 = concatenate([conv1,up13], axis = -1)
    drop13 = Dropout(0.2)(merge13)
    conv13 = Conv1D(8, 3, activation = 'relu', padding = 'same')(drop13)
    conv13 = Conv1D(8, 3, activation = 'relu', padding = 'same')(conv13)
    
    conv14 = Conv1D(1, 1, activation = 'sigmoid')(conv13)

    model = Model(inputs = inputs, outputs = conv14)
    model.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mae', 'acc'])
    model.summary()

    return model

####

'''
def EncoderDecoder(input_size, pool_size):

    inputs = Input(input_size)

    conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_1')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_2')(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=(pool_size))(bn1)

    conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_1')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_2')(conv2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=(pool_size))(bn2)

    conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3_1')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3_2')(conv3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=(pool_size))(bn3)

    conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_1')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_2')(bn4)
    bn4 = BatchNormalization()(conv4)

    pool4 = MaxPooling1D(pool_size=(pool_size))(conv4)

    conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5_1')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5_2')(conv5)
    bn5 = BatchNormalization()(conv5)
    
    #drop5 = Dropout(0.5)(conv5)

    up6 = Conv1D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (pool_size))(bn5))
    #merge6 = concatenate([conv4,up6], axis = -1)
    drop6 = Dropout(0.2)(up6)
    conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop6)
    conv6 = Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv1D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (pool_size))(conv6))
    #merge7 = concatenate([conv3,up7], axis = -1)
    drop7 = Dropout(0.2)(up7)
    conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop7)
    conv7 = Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv1D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (pool_size))(conv7))
    #merge8 = concatenate([conv2,up8], axis = -1)
    drop8 = Dropout(0.2)(up8)
    conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop8)
    conv8 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv1D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(size = (pool_size))(conv8))
    #merge9 = concatenate([conv1,up9], axis = -1)
    drop9 = Dropout(0.2)(up9)
    conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop9)
    conv9 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv1D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv1D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mae', 'acc'])
    model.summary()

    return model    
'''