# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 20:13:36 2021

@author: Omer
"""

import os, sys

import numpy as np
import tensorflow as tf
#tf.enable_eager_execution ()
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, LSTM, LayerNormalization
from tensorflow.keras.layers import Conv2DTranspose, Input, Concatenate, GRU, Add, Reshape, TimeDistributed, Activation
from keras.models import Model
import keras.backend as K
#import neural_structured_learning as nsl
from spektral.layers import GCNConv, GlobalSumPool, ARMAConv, GatedGraphConv,GATConv, MinCutPool
from spektral.layers import APPNPConv, GCSConv
from spektral.models import GCN
import spektral as spek
from spektral.utils import normalized_adjacency
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from tqdm import tqdm
import scipy.io
from sklearn.covariance import GraphicalLassoCV
from numpy import genfromtxt


print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#First few basic methods from online source codes 
def create_dataset(X, look_back=3):
    dataX, dataY = [], []
    for i in tqdm(range(len(X)-look_back-1), desc="Create Dataset"):
         a = X[i:(i+look_back), 0]
         dataX.append(a)
         dataY.append(X[i + look_back, 0])
   
    return np.array(dataX), np.array(dataY)   

def create_dataset2(X, look_back=3):
    dataX, dataY = [], []
    d1,d2 = X.shape 
    x_return = np.zeros((d2,len(X)-look_back-1,look_back))
    y_return = np.zeros((d2,len(X)-look_back-1))

    for j in tqdm(range(X.shape[1]) , desc="Create Dataset"	):
        for i in range(len(X)-look_back-1):
            a = X[i:(i+look_back), j]
            dataX.append(a)
            dataY.append(X[i + look_back, j])
        
        x_return[j] = np.array(dataX)
        y_return[j] = np.array(dataY)
        dataX, dataY = [], []

    return x_return, y_return


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


def total_return_loss(y_true,y_pred):  
    loss = tf.reduce_sum(-y_true*y_pred, axis=1)
    return K.mean(loss)

def sharpe_loss(y_true,y_pred):  
    mean = K.mean( K.sum(-y_true*y_pred , axis=1) )
#    mean = K.mean( tf.reduce_sum(-y_true*y_pred , axis=1) )
    std = K.std( K.sum(-y_true*y_pred, axis=1) )
#    std = K.std( tf.reduce_sum(-y_true*y_pred, axis=1) )
    loss = tf.math.divide(mean,std)
    return loss #K.exp(-loss)

def sharpe_reg_loss(y_true,y_pred):  
    std = K.std( K.sum(-y_true*y_pred, axis=1) )
    loss = tf.reduce_sum((-y_true*y_pred) + (0.01*std), axis=1)     
    return loss

class GatedLinearUnit(layers.Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

#%% JACOPO DATA

path = r'C:\Users\Asus\Desktop\cvxproject\fin data\Datasets'
os.chdir(path)

mat = scipy.io.loadmat('DowJones2005Ret.mat')
X1 = np.array(mat['RR'])

mat = scipy.io.loadmat('EuroBondsRet.mat')
X2 = np.array(mat['RR'])

mat = scipy.io.loadmat('ItBondsandCommoditiesMixRet.mat')
X3 = np.array(mat['RR'])

mat = scipy.io.loadmat('WorldMixBondsRet.mat')
X4 = np.array(mat['RR'])

#X = np.concatenate([X1,X2,X3,X4],axis=1)
X = X2


#%% Mini Partition Fcn

Xmini1 = X[:400,:]
Xmini2 = X[:500,:]
Xmini3 = X[:600,:]
Xmini4 = X[:700,:]
Xmini5 = X[:800,:]
Xmini6 = X[:900,:]
Xmini7 = X[:1000,:]
Xmini8 = X[:1100,:]
Xmini9 = X[:1200,:]
Xmini10 = X[:1300,:]
Xmini11 = X[:1400,:]
Xmini12 = X[:1500,:]

#for i in range(Xmini1.shape[1]):
#    Xmini1[:,i] = ( Xmini1[:,i] - np.mean(Xmini1,axis=0)[i] ) / np.std(Xmini1,axis=0)[i]
#    Xmini2[:,i] = ( Xmini2[:,i] - np.mean(Xmini2,axis=0)[i] ) / np.std(Xmini2,axis=0)[i]
#    Xmini3[:,i] = ( Xmini3[:,i] - np.mean(Xmini3,axis=0)[i] ) / np.std(Xmini3,axis=0)[i]
#    Xmini4[:,i] = ( Xmini4[:,i] - np.mean(Xmini4,axis=0)[i] ) / np.std(Xmini4,axis=0)[i]
#    Xmini5[:,i] = ( Xmini5[:,i] - np.mean(Xmini5,axis=0)[i] ) / np.std(Xmini5,axis=0)[i]
#    Xmini6[:,i] = ( Xmini6[:,i] - np.mean(Xmini6,axis=0)[i] ) / np.std(Xmini6,axis=0)[i]
#    Xmini7[:,i] = ( Xmini7[:,i] - np.mean(Xmini7,axis=0)[i] ) / np.std(Xmini7,axis=0)[i]
#    Xmini8[:,i] = ( Xmini8[:,i] - np.mean(Xmini8,axis=0)[i] ) / np.std(Xmini8,axis=0)[i]
#    Xmini9[:,i] = ( Xmini9[:,i] - np.mean(Xmini9,axis=0)[i] ) / np.std(Xmini9,axis=0)[i]
#    Xmini10[:,i] = ( Xmini10[:,i] - np.mean(Xmini10,axis=0)[i] ) / np.std(Xmini10,axis=0)[i]
#    Xmini11[:,i] = ( Xmini11[:,i] - np.mean(Xmini11,axis=0)[i] ) / np.std(Xmini11,axis=0)[i]
#    Xmini12[:,i] = ( Xmini12[:,i] - np.mean(Xmini12,axis=0)[i] ) / np.std(Xmini12,axis=0)[i]


look_back = 5   #?????????
ind_no = X.shape[1] #int(feature_no/2)
feature_no = ind_no

def GetMiniData(data,type='feat'):
    X_all, Y_all = split_sequences(data,look_back)    
    
    dim1, dim2, feature_no = X_all.shape
    train_size = int(np.floor(dim1*0.8))
    x_train = np.zeros((train_size,dim2,feature_no))    
    
    x_train = X_all[:train_size,:,:]
    x_test = X_all[train_size:,:,:]
    y_train = Y_all[:train_size,:]
    y_test = Y_all[train_size:,:]
    
    #GRAPH DATA FEATURES WITH MEAN AND VARIANCE/STD THUS DIM 2
    x_train_feat = np.zeros((x_train.shape[0],x_train.shape[2],2+look_back))
    x_test_feat = np.zeros((x_test.shape[0],x_train.shape[2],2+look_back))
    
    for i in range(x_train.shape[0]):
        x_train_feat[i,:,0] = np.mean(x_train[i],axis=0)
        x_train_feat[i,:,1] = np.std(x_train[i],axis=0)
        
    for i in range(x_test.shape[0]):
        x_test_feat[i,:,0] = np.mean(x_test[i],axis=0)
        x_test_feat[i,:,1] = np.std(x_test[i],axis=0)
       
    x_train_feat[:,:,2:] = np.swapaxes(x_train,1,2)    
    x_test_feat[:,:,2:] = np.swapaxes(x_test,1,2)
    
    if(type=='feat'):
        return x_train,x_train_feat,x_test,x_test_feat, y_train, y_test
    else:
        return x_train,x_test,y_train,y_test
    
def GetAdjMatrix(data):
    X_cov = (1/data.shape[0])*data.T@data
    X_graph = GraphicalLassoCV()
    X_graph.fit(X_cov)
    
#    cov = X_graph.covariance_
    prec = X_graph.precision_
    
    prec = prec/np.max(prec)
    prec = np.where(prec!=0,1,0)
    
    #NORMALIZE THE ADJ MATRIX
    prec_normal = normalized_adjacency(prec, symmetric=True)
    
    inp_lap_train = np.zeros((x_train.shape[0],prec.shape[0],prec.shape[1]))
    inp_lap_test = np.zeros((x_test.shape[0],prec.shape[0],prec.shape[1]))
    
    for i in range(x_train.shape[0]):
        inp_lap_train[i] = prec
    for i in range(x_test.shape[0]):
        inp_lap_test[i] = prec
        
    return inp_lap_train, inp_lap_test


#%% MODELS!!!

#def FinNN():
#    inpt = Input(shape=(look_back,feature_no), name='input')
#    x = BatchNormalization()(inpt)
#    x = LSTM(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
##    x = LSTM(128, activation='relu')(x)
##    x = Dense(128,activation='relu')(x)
##    x = LayerNormalization()(x) 
#    x = Dense(64,activation='relu')(x)
##    x = GatedLinearUnit(128)(x)
#    out = Dense(ind_no , activation='softmax')(x)
#
#    model = Model(inputs=inpt, outputs=out)
#    opt = keras.optimizers.Adam(learning_rate=0.0001)
#    model.compile(optimizer=opt, loss=total_return_loss)
##    model.compile(optimizer=opt, loss=sharpe_loss)
#    return model
    
def FinNN():
    inpt = Input(shape=(look_back,feature_no), name='input')
    x = BatchNormalization()(inpt)
    x = LSTM(128, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    xx = Dropout(0.3)(x)
    xx = Dense(64, activation='gelu')(xx)
    xx = Concatenate()([x,xx])
    xx = Dropout(0.3)(xx)
    xx = Dense(32, activation='gelu')(xx)
    xx = Dense(ind_no, activation='gelu')(xx)
    x = Concatenate()([x,xx])
    out = Dense(ind_no, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=out)
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
#    model.compile(optimizer=opt, loss=sharpe_loss)
    return model

def FinNN2():
    inpt = Input(shape=(look_back+2,feature_no), name='input')
    x = BatchNormalization()(inpt)
    x = LSTM(128, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    xx = Dropout(0.3)(x)
    xx = Dense(64, activation='gelu')(xx)
    xx = Concatenate()([x,xx])
    xx = Dropout(0.3)(xx)
    xx = Dense(32, activation='gelu')(xx)
    xx = Dense(ind_no, activation='gelu')(xx)
    x = Concatenate()([x,xx])
    out = Dense(ind_no, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=out)
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

#def FinNN2():
#    inpt = Input(shape=(look_back,feature_no), name='input')
#    x = BatchNormalization()(inpt)
#    x = LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))(x)
#    x = TimeDistributed(Dense(1))(x)
#    x = Flatten()(x)
#    out = Dense(ind_no , activation='softmax')(x)
#
#    model = Model(inputs=inpt, outputs=out)
#    opt = keras.optimizers.Adam(learning_rate=0.0001)
#    model.compile(optimizer=opt, loss=total_return_loss)
#    return model
    
#???
def FinNN_Multi():
    inpt = Input(shape=(look_back,feature_no), name='input')
    x = BatchNormalization()(inpt)
    x = LSTM(64, activation='gelu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
#    x = LSTM(128, activation='relu')(x)
#    x = LayerNormalization()(x) 
    x = Dense(64,activation='gelu')(x)
    out_ret = Dense(ind_no , activation='softmax',name='return')(x)
    out_sharpe = Dense(ind_no , activation='softmax',name='sharpe')(x)

    model = Model(inputs=inpt, outputs=[out_ret, out_sharpe])
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
#    opt = keras.optimizers.Adam(learning_rate=0.0001, decay=0.0001 / 200) 

    losses = {"return": total_return_loss, "sharpe": sharpe_loss}
    lossWeights = {"return": 1.0, "sharpe": 1.0}

    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights )
    
    return model


def GraphModel():
    inp_seq = Input(shape=(look_back, feature_no),name='inp_seq')
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GCSConv(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4))([inp_feat, inp_lap])
    x = GCSConv(1, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4))([inp_feat, inp_lap])
#    x = ARMAConv(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3))([inp_feat, inp_lap])
#    x = GCNConv(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3))([inp_feat, inp_lap])
#    x = GATConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4))([x, inp_lap])
    x = Flatten()(x)
    
    xx = BatchNormalization()(inp_seq)
    xx = LSTM(128, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(xx) #,return_sequences=True
#    xx = LSTM(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5))(xx)
    x = Concatenate()([x,xx])
#    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
#    x = Dropout(0.3)(x)
#    x = Dense(32, activation='relu')(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_seq, inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def GraphModel_old():
    inp_seq = Input(shape=(look_back, feature_no),name='inp_seq')
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GCNConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))([inp_feat, inp_lap])
    x = GCNConv(16, activation='relu')([x, inp_lap])
    x = Flatten()(x)
    xx = LSTM(128, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(inp_seq) #,return_sequences=True
#    xx = LSTM(16, activation='relu')(xx)
    x = Concatenate()([x,xx])
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_seq, inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_GCN():
    inp_seq = Input(shape=(look_back, feature_no),name='inp_seq')
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GCNConv(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))([inp_feat, inp_lap])
    x = GCNConv(32, activation='relu')([x, inp_lap])
    x = Flatten()(x)
#    x = Dense(128, activation='relu')(x)
    x = GatedLinearUnit(32)(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_seq, inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_ARMA():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = ARMAConv(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))([inp_feat, inp_lap])
    x = ARMAConv(128, activation='relu')([x, inp_lap])
    x = ARMAConv(1)([x, inp_lap])
    x = TimeDistributed(Dense(1))(x)

    x = Flatten()(x)
    xx = Dropout(0.3)(x)
    xx = Dense(64, activation='gelu')(xx)
    xx = Concatenate()([x,xx])
    xx = Dropout(0.3)(xx)
    xx = Dense(32, activation='gelu')(xx)
    xx = Dense(ind_no, activation='gelu')(xx)
    x = Concatenate()([x,xx])
    out = Dense(ind_no, activation='softmax')(x)
#    x = GatedLinearUnit(32)(x)
#    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[ inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

#???
def Graph_GAT():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GATConv(64,attn_heads=4, activation='gelu',kernel_regularizer=regularizers.l1_l2(l1=5e-7, l2=1e-6))([inp_feat, inp_lap])
    x = GATConv(128,attn_heads=2, activation='gelu')([x, inp_lap])
    x = GATConv(1,attn_heads=1)([x, inp_lap])
        
    x = Flatten()(x)
    xx = Dropout(0.3)(x)
    xx = Dense(64, activation='gelu')(xx)
    xx = Concatenate()([x,xx])
    xx = Dropout(0.3)(xx)
    xx = Dense(32, activation='gelu')(xx)
    xx = Dense(ind_no, activation='gelu')(xx)
    x = Concatenate()([x,xx])
    out = Dense(ind_no, activation='softmax')(x)

#    xx = GatedLinearUnit(32)(xx)
    
    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
#    model.compile(optimizer=opt, loss=sharpe_loss)
    return model

#ORIGINAL ONE
#def Graph_GAT():
#    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
#    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
#    x = GATConv(64,attn_heads=4, activation='gelu',kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-6))([inp_feat, inp_lap])
#    x = GATConv(128,attn_heads=2, activation='gelu')([x, inp_lap])
#    x = GATConv(1,attn_heads=1)([x, inp_lap])
#    
#    x = TimeDistributed(Dense(1))(x)
#    
#    x = Flatten()(x)
##    x = BatchNormalization()(x)
##    x = GatedLinearUnit(32)(x)
#    x = Dense(128,activation='gelu')(x)
#    x = Dense(32,activation='gelu')(x)
#    out = Dense(ind_no, activation='softmax')(x)
#    
#    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
#    
#    opt = keras.optimizers.Adam(learning_rate=0.0001)
#    model.compile(optimizer=opt, loss=total_return_loss)
##    model.compile(optimizer=opt, loss=sharpe_loss)
#    return model

def Graph_GAT_Multi():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GATConv(64,attn_heads=4, activation='gelu',kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-6))([inp_feat, inp_lap])
    x = GATConv(128,attn_heads=2, activation='gelu')([x, inp_lap])
    x = GATConv(1,attn_heads=1)([x, inp_lap])
    
#    x = TimeDistributed(Dense(1))(x)
    
    x = Flatten()(x)
    x = BatchNormalization()(x)
#    x = GatedLinearUnit(32)(x)
    x = Dense(64,activation='gelu')(x)    
    out_ret = Dense(ind_no , activation='softmax',name='return')(x)
    out_sharpe = Dense(ind_no , activation='softmax',name='sharpe')(x)

    model = Model(inputs=[inp_lap, inp_feat], outputs=[out_ret, out_sharpe])
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
#    opt = keras.optimizers.Adam(learning_rate=0.0001, decay=0.0001 / 200) 

    losses = {"return": total_return_loss, "sharpe": sharpe_loss}
    lossWeights = {"return": 1.0, "sharpe": 1.0}

    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights )
    
    return model

def Graph_APPNP():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = APPNPConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=5e-6, l2=1e-5))([inp_feat, inp_lap])
    x = APPNPConv(64, activation='relu')([x, inp_lap])
    x = APPNPConv(1)([x, inp_lap])
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model


#???
#def Graph_GCS():
#    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
#    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
#    x = GCSConv(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=5e-7, l2=1e-5))([inp_feat, inp_lap])
#    x = GCSConv(128, activation='relu')([x, inp_lap])
#    x = GCSConv(1, activation='relu')([x, inp_lap])
#    x = Flatten()(x)
#    xx = Dropout(0.3)(x)
#    xx = BatchNormalization()(xx) 
#    xx = GatedLinearUnit(64)(xx)  
#    xx = Concatenate()([x,xx])
#    xx = Dropout(0.3)(xx)
#    xx = BatchNormalization()(xx) 
#    x = GatedLinearUnit(32)(x) 
#    out = Dense(ind_no, activation='softmax')(x)
#    
#    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
#    
#    opt = keras.optimizers.Adam(learning_rate=0.0005)
#    model.compile(optimizer=opt, loss=total_return_loss)
#    return model

def Graph_GCS():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GCSConv(64, activation='gelu',kernel_regularizer=regularizers.l1_l2(l1=5e-7, l2=1e-5))([inp_feat, inp_lap])
    x = GCSConv(128, activation='gelu')([x, inp_lap])
    x = GCSConv(1, activation='gelu')([x, inp_lap])
    x = Flatten()(x)
    xx = Dropout(0.3)(x)
    xx = Dense(64, activation='gelu')(xx)
    xx = Concatenate()([x,xx])
    xx = Dropout(0.3)(xx)
    xx = Dense(32, activation='gelu')(xx)
    xx = Dense(ind_no, activation='gelu')(xx)
    x = Concatenate()([x,xx])
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_MIX():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GCSConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4))([inp_feat, inp_lap])
    x = GCSConv(64, activation='relu')([x, inp_lap])
    x = GCSConv(1)([x, inp_lap])
#    x = Flatten()(x)
    
    xx = ARMAConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4))([inp_feat, inp_lap])
    xx = ARMAConv(64, activation='relu')([xx, inp_lap])
    xx = ARMAConv(1)([xx, inp_lap])
#    xx = Flatten()(xx)
    
#    x = Concatenate()([x,xx])
    x = Add()([x,xx])
    x = Flatten()(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
#    x = Dense(32, activation='relu')(x)    
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model
    
def Graph_MIX2():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GATConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4))([inp_feat, inp_lap])
    x = GATConv(64, activation='relu')([x, inp_lap])
    x = GCSConv(128, activation='relu')([inp_feat, inp_lap])
    x = GATConv(1)([x, inp_lap])
    x = Flatten()(x)
        
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_MIX3():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    x = GCSConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4))([inp_feat, inp_lap])
    x = GCSConv(64, activation='relu')([x, inp_lap])
    x = GCSConv(128, activation='relu')([x, inp_lap])
    x = GCSConv(1)([x, inp_lap])
    x = Flatten()(x)
        
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_GraLSTM():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    
    x = GATConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))([inp_feat, inp_lap])
    x = GATConv(64, activation='relu')([x, inp_lap])
    x = GATConv(128, activation='relu')([x, inp_lap])
    x = GATConv(1, activation='relu')([x, inp_lap])
#    x = GATConv(look_back)([x, inp_lap])
#    x = Flatten()(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = BatchNormalization()(x)

    x = LSTM(128, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))(x)
    x = GatedLinearUnit(128)(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_GraLSTM2():
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')
    
    x = GATConv(32, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))([inp_feat, inp_lap])
    x = GCSConv(64, activation='relu')([x, inp_lap])
    x = GATConv(128, activation='relu')([x, inp_lap])
#    x = GATConv(look_back)([x, inp_lap])
    x = GATConv(1, activation='relu')([x, inp_lap])

    #MAYBE ADD 1 filter layer
    x = tf.transpose(x, perm=[0, 2, 1])

    x = BatchNormalization()(x)

    x = GRU(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_FEAT():
    inp_seq = Input(shape=(look_back, feature_no),name='inp_seq')
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    x = BatchNormalization()(inp_seq)

    x = LSTM(128, activation='relu',return_sequences=True,kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))(x)
    x = LSTM(ind_no, activation='relu')(x)    
    x = Reshape((-1, 1))(x)
    x = GATConv(64, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))([x, inp_lap])
    x = GATConv(128, activation='relu')([x, inp_lap])
    x = GATConv(1, activation='relu')([x, inp_lap])
    x = tf.transpose(x, perm=[0, 2, 1])
    x = TimeDistributed(Dense(ind_no))(x)    #ind no or 1
    x = Flatten()(x)
#    x = Dense(128, activation='relu')(x)
    x = GatedLinearUnit(64)(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_seq], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_FEAT2():
    inp_seq = Input(shape=(look_back, feature_no),name='inp_seq')
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')

    x = LSTM(128, activation='relu',return_sequences=True,kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))(inp_seq)
    x = LSTM(ind_no, activation='relu')(x)    
    x = Reshape((-1, 1))(x)
    x = GATConv(256, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-5))([x, inp_lap])
    x = Flatten()(x)    
#    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
#    x = Dropout(0.3)(x)
    out = Dense(ind_no, activation='softmax')(x)
    
    model = Model(inputs=[inp_lap, inp_seq], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=total_return_loss)
    return model

def Graph_FEAT_Concat():
    inp_seq = Input(shape=(look_back, feature_no),name='inp_seq')
    inp_lap = Input(shape=(feature_no, feature_no),name='inp_lap')
    inp_feat = Input(shape=(feature_no, x_train_feat.shape[-1]),name='inp_feat')

    x = LSTM(128, activation='relu',return_sequences=True,kernel_regularizer=regularizers.l1_l2(l1=1e-7, l2=1e-7))(inp_seq)
    x = LSTM(ind_no, activation='relu')(x)    
    x = Reshape((-1, 1))(x)
    x = Concatenate()([x,inp_feat])
    x = GCSConv(128,activation='relu')([x, inp_lap])
    x = GCSConv(32,activation='relu')([x, inp_lap])
    x = GCSConv(1)([x, inp_lap])
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)

    out = Dense(ind_no, activation='softmax')(x)
    model = Model(inputs=[inp_lap, inp_seq,inp_feat], outputs=out)
    
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss=sharpe_loss)

    return model



#%% FUNCTIONS FOR TEST

def GraphTest(name,verb=2):
    if(name == 'GAT'):
        gmodel = Graph_GAT()
        epoch_no = 600
    elif(name == 'ARMA'):
        gmodel = Graph_ARMA()
        epoch_no = 600
    elif(name == 'APPNP'):
        gmodel = Graph_APPNP()
        epoch_no = 300
    elif(name == 'GCS'):
        gmodel = Graph_GCS()
        epoch_no = 700
    elif(name == 'GCN'):
        gmodel = Graph_GCN()
        epoch_no = 1200
    elif(name == 'MIX'):
        gmodel = Graph_MIX()
        epoch_no = 1200
    elif(name == 'MIX2'):
        gmodel = Graph_MIX2()
        epoch_no = 500
    elif(name == 'MIX3'):
        gmodel = Graph_MIX3()
        epoch_no = 1200
    elif(name == 'multi'):
        gmodel = Graph_GAT_Multi()
        epoch_no = 500
    else:
        print('no graph specified')

    ghistory = gmodel.fit([inp_lap_train,x_train_feat],y_train,epochs=epoch_no, validation_split=0.1,verbose=verb)
    res = gmodel.predict([inp_lap_test,x_test_feat])
    
    if(name != 'multi'):
        
        ret_gnn = np.sum(y_test*res , axis = 1) #r_test or y_test

        total_reward = np.sum(y_test*res) 
        print('total reward is = ', total_reward)
        
        mean_ret_gnn = np.mean(ret_gnn)
        std_ret_gnn = np.std(ret_gnn)
        sharpe_ret_gnn = mean_ret_gnn/std_ret_gnn
        
        print('mean reward is = ', mean_ret_gnn)
        print('std reward is = ', std_ret_gnn )
        print('sharpe ratio is = ', sharpe_ret_gnn )
    else:
        
        ret_gnn = np.sum(y_test*res[0] , axis = 1) #r_test or y_test
        cum_ret_gnn = np.cumsum(ret_gnn)
                
        total_reward = np.sum(y_test*res[0]) 
        print('total reward is = ', total_reward)
        
        mean_ret_gnn = np.mean(ret_gnn)
        std_ret_gnn = np.std(ret_gnn)
        sharpe_ret_gnn = mean_ret_gnn/std_ret_gnn
        
        print('mean reward is = ', mean_ret_gnn)
        print('std reward is = ', std_ret_gnn )
        print('sharpe ratio is = ', sharpe_ret_gnn )
        
        res = res[0]
        
        
    return ghistory,res,total_reward,sharpe_ret_gnn


def GraphFeatTest(name,verb=2):
    if(name=='CONCAT'):
        gmodel = Graph_FEAT_Concat()
        epoch_no = 1500
        ghistory = gmodel.fit([inp_lap_train,x_train,x_train_feat],y_train,epochs=epoch_no, validation_split=0.1,verbose=verb)
        res = gmodel.predict([inp_lap_test,x_test,x_test_feat])
    elif(name=='GRALSTM'):
        gmodel = Graph_GraLSTM()
        epoch_no = 400
        ghistory = gmodel.fit([inp_lap_train,x_train_feat],y_train,epochs=epoch_no, validation_split=0.1,verbose=verb)
        res = gmodel.predict([inp_lap_test,x_test_feat])
    elif(name=='GRALSTM2'):
        gmodel = Graph_GraLSTM2()
        epoch_no = 400
        ghistory = gmodel.fit([inp_lap_train,x_train_feat],y_train,epochs=epoch_no, validation_split=0.1,verbose=verb)
        res = gmodel.predict([inp_lap_test,x_test_feat])   
    else:
        gmodel = Graph_FEAT()
        epoch_no = 1500
        ghistory = gmodel.fit([inp_lap_train,x_train],y_train,epochs=epoch_no, validation_split=0.1,verbose=verb)
        res = gmodel.predict([inp_lap_test,x_test])
       
#    ghistory = gmodel.fit([inp_lap_train,x_train,x_train_feat],y_train,epochs=epoch_no, validation_split=0.1,verbose=verb)
#    res = gmodel.predict([inp_lap_test,x_test,x_test_feat])
    
    ##%%TRAIN LOSS PLOT
#    plt.figure()
#    plt.plot(ghistory.history['loss'],'b',label='train loss')
#    plt.plot(ghistory.history['val_loss'],'r',label='valid loss')
#    title = 'Training Loss for ' + name
#    plt.title(title)
#    plt.legend(loc="upper right")
    
    
    ret_gnn = np.sum(y_test*res , axis = 1) #r_test or y_test
    cum_ret_gnn = np.cumsum(ret_gnn)
    

#    plt.figure()
#    title = 'CumSum Returns for ' + name
#    plt.title(title)
#    plt.plot(cum_ret_gnn)
    
    total_reward = np.sum(y_test*res) 
    print('total reward is = ', total_reward)
    
    mean_ret_gnn = np.mean(ret_gnn)
    std_ret_gnn = np.std(ret_gnn)
    sharpe_ret_gnn = mean_ret_gnn/std_ret_gnn
    
    print('mean reward is = ', mean_ret_gnn)
    print('std reward is = ', std_ret_gnn )
    print('sharpe ratio is = ', sharpe_ret_gnn )
    
    return ghistory,res,total_reward,sharpe_ret_gnn

    
#???
def LSTMTest(name,verb=2):
    if(name == '1'):
        model = FinNN() 
        history = model.fit(x_train, y_train, epochs=500 , validation_split=0.1,verbose=verb)
        res = model.predict(x_test)
    elif(name == '2'):
        model = FinNN2() 
        history = model.fit(np.swapaxes(x_test_feat,1,2), y_train, epochs=500 , validation_split=0.1,verbose=verb)
        res = model.predict(np.swapaxes(x_test_feat,1,2))
    elif(name == 'multi'):
        model = FinNN_Multi()
        history = model.fit(x_train, y=y_train, epochs=500 , validation_split=0.1,verbose=verb)
        res = model.predict(x_test)
      
    if(name == '1' or name == '2'):
    #    plt.figure()
    #    plt.plot(history.history['loss'],'b',label='train loss')
    #    plt.plot(history.history['val_loss'],'r',label='valid loss')
    #    title = 'Training Loss for LSTM ' + name
    #    plt.title(title)
    #    plt.legend(loc="upper right")
        
        
        ret_nn = np.sum(y_test*res , axis = 1) #r_test or y_test
        cum_ret_nn = np.cumsum(ret_nn)
        
    
    #    plt.figure()
    #    title = 'CumSum Returns for LSTM ' + name
    #    plt.title(title)
    #    plt.plot(cum_ret_nn)
        
        total_reward = np.sum(y_test*res) 
        print('total reward is = ', total_reward)
        
        mean_ret_nn = np.mean(ret_nn)
        std_ret_nn = np.std(ret_nn)
        sharpe_ret_nn = mean_ret_nn/std_ret_nn
        
        print('mean reward is = ', mean_ret_nn)
        print('std reward is = ', std_ret_nn )
        print('sharpe ratio is = ', sharpe_ret_nn )
    
    else:
        plt.figure()
        plt.plot(history.history['return_loss'],'b',label='train loss')
        plt.plot(history.history['val_return_loss'],'r',label='valid loss')
        title = 'Training Return Loss for LSTM ' + name
        plt.title(title)
        plt.legend(loc="upper right")
        
        plt.figure()
        plt.plot(history.history['sharpe_loss'],'b',label='train loss')
        plt.plot(history.history['val_sharpe_loss'],'r',label='valid loss')
        title = 'Training Sharpe Loss for LSTM ' + name
        plt.title(title)
        plt.legend(loc="upper right")
        
        
        ret_nn = np.sum(y_test*res[0] , axis = 1) #r_test or y_test
        cum_ret_nn = np.cumsum(ret_nn)
        
        plt.figure()
        title = 'CumSum Returns for LSTM ' + name
        plt.title(title)
        plt.plot(cum_ret_nn)
        
        total_reward = np.sum(y_test*res[0]) 
        print('total reward is = ', total_reward)
        
        mean_ret_nn = np.mean(ret_nn)
        std_ret_nn = np.std(ret_nn)
        sharpe_ret_nn = mean_ret_nn/std_ret_nn
        
        print('mean reward is = ', mean_ret_nn)
        print('std reward is = ', std_ret_nn )
        print('sharpe ratio is = ', sharpe_ret_nn )
        
        #??? CHECK THIS
        res = res[0]
    
    return history,res,total_reward,sharpe_ret_nn


def MarkovTest(data):
    dim1,dim2 = data.shape    
    train_size = int(np.floor(dim1*0.8))
    x_train_mark = data[:train_size,:]
    x_test_mark = data[train_size:,:]
    
    gamma = 1
    x_mark = x_train_mark
    mu = np.mean(x_mark,axis=0).reshape(-1,1)
    cov = (1/data.shape[0])*x_mark.T@x_mark
    
    w_mark = (1/gamma) * np.linalg.inv(cov) @ mu
    
    w_mark = w_mark / np.sum(np.abs(w_mark))
    
    ret_mark = x_test_mark@w_mark
#    
#    plt.figure()
#    plt.title('CumSum Returns MARKOVITZ')
#    plt.plot(np.cumsum(ret_mark))
#    
    total_reward = np.sum(x_test_mark@w_mark) 
    print('total reward is = ', total_reward)
    
    mean_ret = np.mean(ret_mark)
    std_ret = np.std(ret_mark)
    sharpe_ret = mean_ret/std_ret
    
    print('mean reward is = ', mean_ret)
    print('std reward is = ', std_ret )
    print('sharpe ratio is = ', sharpe_ret )

    return w_mark,total_reward,sharpe_ret


#%% Mini FEAT Test
#    
#DATA = Xmini11
#
#x_train,x_train_feat,x_test,x_test_feat, y_train, y_test = GetMiniData(DATA,'feat')
#inp_l  ap_train, inp_lap_test = GetAdjMatrix(DATA)
#
##_,_,total_reward1,sharpe_ret1 = LSTMTest('1',0)
##_,_,total_reward2,sharpe_ret2 = GraphFeatTest('GRALSTM',0)
##_,_,total_reward3,sharpe_ret3 = GraphFeatTest('GRALSTM2',0)
##_,resgat,total_reward,sharpe_ret = GraphTest('GAT',1)
#_,resgat,total_reward,sharpe_ret = GraphTest('ARMA',1)
##_,resgcs,total_reward2,sharpe_ret2 = GraphTest('GCS',0)
##_,_,total_reward2,sharpe_ret2 = GraphTest('MIX2',0)
##total_reward, sharpe_ret = MarkovTest(DATA)
#
#
##_,_,total_reward1,sharpe_ret1 = LSTMTest('multi',2)
#
##_,resgat,total_reward,sharpe_ret = GraphTest('multi',0)




#%% Mini Tests
#data_list = [Xmini1,Xmini2,Xmini3,Xmini4,Xmini5,Xmini6,Xmini7,Xmini8,Xmini9,Xmini10,Xmini11,Xmini12]
#MIX_RET = []
#MIX_SHARP = []
#
#iter_no = 0
#sum = 0
#for Xmini in data_list:
#    
#    x_train,x_train_feat,x_test,x_test_feat, y_train, y_test = GetMiniData(Xmini,'feat')
##    inp_lap_train, inp_lap_test = GetAdjMatrix(Xmini)
#    
##    print('MIX TEST BELOW')
##    _,_,total_reward,sharpe_ret = GraphTest('MIX',0)
##    MIX_RET.append(total_reward)
##    MIX_SHARP.append(sharpe_ret)
#    print('Rand TEST BELOW')
#    res_mark,total_reward, sharpe_ret = MarkovTest(Xmini)
#    print(total_reward)
#    sum +=total_reward
#    
#    iter_no += 1
#
#print(sum)

    
#%% FULL TEST 

L_RET1 = []
L_SHARP1 = []
L_RET2 = []
L_SHARP2 = []
L_RET3 = []
L_SHARP3 = []
L_RET4 = []
L_SHARP4 = []
L_RET5 = []
L_SHARP5 = []
L_RET6 = []
L_SHARP6 = []
L_RET7 = []
L_SHARP7 = []
L_RET8 = []
L_SHARP8 = []
L_RET9 = []
L_SHARP9 = []
L_RET10 = []
L_SHARP10 = []




data_list = [Xmini1,Xmini2,Xmini3,Xmini4,Xmini5,Xmini6,Xmini7,Xmini8,Xmini9,Xmini10,Xmini11,Xmini12]

iter_no = 0
for Xmini in data_list:
    print('STARTING ITER ', iter_no )
    x_train,x_train_feat,x_test,x_test_feat, y_train, y_test = GetMiniData(Xmini,'feat')
    inp_lap_train, inp_lap_test = GetAdjMatrix(Xmini)
    
    print('GAT 1 TEST BELOW')
#    _,_,total_reward1,sharpe_ret1 = GraphTest('GAT',0)
#    _,_,total_reward1,sharpe_ret1 = GraphTest('GCS',0)
#    _,_,total_reward1,sharpe_ret1 = LSTMTest('1',0)
    _,_,total_reward1,sharpe_ret1 = GraphTest('ARMA',0)
#    _,total_reward1,sharpe_ret1 = MarkovTest(Xmini)
    L_RET1.append(total_reward1)
    L_SHARP1.append(sharpe_ret1)

    print('GAT 2 TEST BELOW')
#    _,_,total_reward2,sharpe_ret2 = GraphTest('GAT',0)
#    _,_,total_reward2,sharpe_ret2 = GraphTest('GCS',0)
#    _,_,total_reward2,sharpe_ret2 = LSTMTest('1',0)
    _,_,total_reward2,sharpe_ret2 = GraphTest('ARMA',0)
#    _,total_reward2,sharpe_ret2 = MarkovTest(Xmini)
    L_RET2.append(total_reward2)
    L_SHARP2.append(sharpe_ret2)

    print('GAT 3 TEST BELOW')
#    _,_,total_reward3,sharpe_ret3 = GraphTest('GAT',0)
#    _,_,total_reward3,sharpe_ret3 = GraphTest('GCS',0)
#    _,_,total_reward3,sharpe_ret3 = LSTMTest('1',0)
    _,_,total_reward3,sharpe_ret3 = GraphTest('ARMA',0)
#    _,total_reward3,sharpe_ret3 = MarkovTest(Xmini)
    L_RET3.append(total_reward3)
    L_SHARP3.append(sharpe_ret3)

    print('GAT 4 TEST BELOW')
#    _,_,total_reward4,sharpe_ret4 = GraphTest('GAT',0)
#    _,_,total_reward4,sharpe_ret4 = GraphTest('GCS',0)
#    _,_,total_reward4,sharpe_ret4 = LSTMTest('1',0)
    _,_,total_reward4,sharpe_ret4 = GraphTest('ARMA',0)
#    _,total_reward4,sharpe_ret4 = MarkovTest(Xmini)
    L_RET4.append(total_reward4)
    L_SHARP4.append(sharpe_ret4)
    
    print('GAT 5 TEST BELOW')
#    _,_,total_reward5,sharpe_ret5 = GraphTest('GAT',0)
#    _,_,total_reward5,sharpe_ret5 = GraphTest('GCS',0)
#    _,_,total_reward5,sharpe_ret5 = LSTMTest('1',0)
    _,_,total_reward5,sharpe_ret5 = GraphTest('ARMA',0)
#    _,total_reward5,sharpe_ret5 = MarkovTest(Xmini)
    L_RET5.append(total_reward5)
    L_SHARP5.append(sharpe_ret5)
    
    print('GAT 6 TEST BELOW')
#    _,_,total_reward6,sharpe_ret6 = GraphTest('GAT',0)
#    _,_,total_reward6,sharpe_ret6 = GraphTest('GCS',0)
#    _,_,total_reward6,sharpe_ret6 = LSTMTest('1',0)
    _,_,total_reward6,sharpe_ret6 = GraphTest('ARMA',0)
#    _,total_reward6,sharpe_ret6 = MarkovTest(Xmini)
    L_RET6.append(total_reward6)
    L_SHARP6.append(sharpe_ret6)
    
    print('GAT 7 TEST BELOW')
#    _,_,total_reward7,sharpe_ret7 = GraphTest('GAT',0)
#    _,_,total_reward7,sharpe_ret7 = GraphTest('GCS',0)
#    _,_,total_reward7,sharpe_ret7 = LSTMTest('1',0)
    _,_,total_reward7,sharpe_ret7 = GraphTest('ARMA',0)
#    _,total_reward7,sharpe_ret7 = MarkovTest(Xmini)
    L_RET7.append(total_reward7)
    L_SHARP7.append(sharpe_ret7)
    
    print('GAT 8 TEST BELOW')
#    _,_,total_reward8,sharpe_ret8 = GraphTest('GAT',0)
#    _,_,total_reward8,sharpe_ret8 = GraphTest('GCS',0)
#    _,_,total_reward8,sharpe_ret8 = LSTMTest('1',0)
    _,_,total_reward8,sharpe_ret8 = GraphTest('ARMA',0)
#    _,total_reward8,sharpe_ret8 = MarkovTest(Xmini)
    L_RET8.append(total_reward8)
    L_SHARP8.append(sharpe_ret8)
    
    print('GAT 9 TEST BELOW')
#    _,_,total_reward9,sharpe_ret9 = GraphTest('GAT',0)
#    _,_,total_reward9,sharpe_ret9 = GraphTest('GCS',0)
#    _,_,total_reward9,sharpe_ret9 = LSTMTest('1',0)
    _,_,total_reward9,sharpe_ret9 = GraphTest('ARMA',0)
#    _,total_reward9,sharpe_ret9 = MarkovTest(Xmini)
    L_RET9.append(total_reward9)
    L_SHARP9.append(sharpe_ret9)
    
    print('GAT 10 TEST BELOW')
#    _,_,total_reward10,sharpe_ret10 = GraphTest('GAT',0)
#    _,_,total_reward10,sharpe_ret10 = GraphTest('GCS',0)
#    _,_,total_reward10,sharpe_ret10 = LSTMTest('1',0)
    _,_,total_reward10,sharpe_ret10 = GraphTest('ARMA',0)
#    _,total_reward10,sharpe_ret10 = MarkovTest(Xmini)
    L_RET10.append(total_reward10)
    L_SHARP10.append(sharpe_ret10)
    
    iter_no += 1


print('L1', sum(L_RET1))
print('L2', sum(L_RET2))
print('L3', sum(L_RET3))
print('L4', sum(L_RET4))
print('L5', sum(L_RET5))
print('L6', sum(L_RET6))
print('L7', sum(L_RET7))
print('L8', sum(L_RET8))
print('L9', sum(L_RET9))
print('L10', sum(L_RET10))

print('L1', sum(L_SHARP1))
print('L2', sum(L_SHARP2))
print('L3', sum(L_SHARP3))
print('L4', sum(L_SHARP4))
print('L5', sum(L_SHARP5))
print('L6', sum(L_SHARP6))
print('L7', sum(L_SHARP7))
print('L8', sum(L_SHARP8))
print('L9', sum(L_SHARP9))
print('L10', sum(L_SHARP10))


#%%

save_lists1 = np.asarray(L_RET1).reshape(-1,1)
save_lists2 = np.asarray(L_RET2).reshape(-1,1)
save_lists3 = np.asarray(L_RET3).reshape(-1,1)
save_lists4 = np.asarray(L_RET4).reshape(-1,1)
save_lists5 = np.asarray(L_RET5).reshape(-1,1)
save_lists6 = np.asarray(L_RET6).reshape(-1,1)
save_lists7 = np.asarray(L_RET7).reshape(-1,1)
save_lists8 = np.asarray(L_RET8).reshape(-1,1)
save_lists9 = np.asarray(L_RET9).reshape(-1,1)
save_lists10 = np.asarray(L_RET10).reshape(-1,1)

save_array = np.concatenate((save_lists1, save_lists2, save_lists3, save_lists4, save_lists5, save_lists6, save_lists7, save_lists8, save_lists9, save_lists10),axis=1)

#np.savetxt('save_array_gat.csv', save_array, delimiter=',')
#np.savetxt('save_array_gcs3.csv', save_array, delimiter=',')
np.savetxt('save_array_arma22.csv', save_array, delimiter=',')
#np.savetxt('save_array_LSTMN2.csv', save_array, delimiter=',')


#%% LOAD AND PLOT

load_array_gat = np.loadtxt('save_array_gat4.csv', delimiter=',')

load_array_arma = np.loadtxt('save_array_arma4.csv', delimiter=',')

load_array_gcs = np.loadtxt('save_array_gcs4.csv', delimiter=',')

load_array_lstm = np.loadtxt('save_array_lstm4.csv', delimiter=',')

#load_array_lstm = np.loadtxt('save_array_LSTMN2.csv', delimiter=',')


#meanresult = np.mean(np.sum(load_array_gat,axis=0))
#meansharpe= (sum(L_SHARP1)+sum(L_SHARP2)+sum(L_SHARP3)+sum(L_SHARP4)+sum(L_SHARP6)+sum(L_SHARP5)+sum(L_SHARP7)+sum(L_SHARP8)+sum(L_SHARP9)+sum(L_SHARP10))/10
#print('mean result = ', meanresult)

#%% Return Stats for Every Instance

means1 = np.mean(load_array_gat,axis=1)
stds1 = np.std(load_array_gat,axis=1)
means2 = np.mean(load_array_lstm,axis=1)
stds2 = np.std(load_array_lstm,axis=1)
means3 = np.mean(load_array_gcs,axis=1)
stds3 = np.std(load_array_gcs,axis=1)
means4 = np.mean(load_array_arma,axis=1)
stds4 = np.std(load_array_arma,axis=1)


print('gat means')
for i in range(12):
    print(np.round(means1[i],6))

print('gat stds') 
for i in range(12):
    print(np.round(stds1[i],6))
    
means1 = np.mean(load_array_gat,axis=1)
stds1 = np.std(load_array_gat,axis=1)

print('LSTM means')
for i in range(12):
    print(np.round(means2[i],6))

print('LSTM stds') 
for i in range(12):
    print(np.round(stds2[i],6))
    
means1 = np.mean(load_array_gat,axis=1)
stds1 = np.std(load_array_gat,axis=1)

print('GCS means')
for i in range(12):
    print(np.round(means3[i],6))

print('GCS stds') 
for i in range(12):
    print(np.round(stds3[i],6))
    
means1 = np.mean(load_array_gat,axis=1)
stds1 = np.std(load_array_gat,axis=1)

print('ARMA means')
for i in range(12):
    print(np.round(means4[i],6))

print('ARMA stds') 
for i in range(12):
    print(np.round(stds4[i],6))


#%% PLOT GAT

#errs = np.zeros((load_array_gat.shape[0]))
#means = np.zeros((load_array_gat.shape[0]))
#stds = np.zeros((load_array_gat.shape[0]))
#for i in range(load_array_gat.shape[0]):
#    errs[i] = np.abs( np.max(load_array_gat[i]) - np.min(load_array_gat[i]) )
#    means[i] = np.mean( load_array_gat[i] )
#    stds[i] = np.std( load_array_gat[i] )
#x = [1,2,3,4,5,6,7,8,9,10,11,12]
#
#plt.figure()
#plt.errorbar(x,means,yerr=stds,fmt='o',ecolor = 'cyan',color='black')
#plt.xlabel('Instance')
#plt.ylabel('Return')
#plt.title('GAT RETURNS')
##plt.savefig("GAT_RETURNS_PLOT4_std.pdf")

#%% PLOT GCS

#errs = np.zeros((load_array_gat.shape[0]))
#means = np.zeros((load_array_gat.shape[0]))
#stds = np.zeros((load_array_gat.shape[0]))
#for i in range(load_array_gat.shape[0]):
#    errs[i] = np.abs( np.max(load_array_gat[i]) - np.min(load_array_gat[i]) )
#    means[i] = np.mean( load_array_gat[i] )
#    stds[i] = np.std( load_array_gat[i] )
#x = [1,2,3,4,5,6,7,8,9,10,11,12]
#
#plt.figure()
#plt.errorbar(x,means,yerr=stds,fmt='o',ecolor = 'cyan',color='black')
#plt.xlabel('Instance')
#plt.ylabel('Return')
#plt.title('GCS RETURNS')
#plt.savefig("GCS_RETURNS_PLOT3_std.pdf")

#%% PLOT ARMA

errs = np.zeros((load_array_gat.shape[0]))
means = np.zeros((load_array_gat.shape[0]))
stds = np.zeros((load_array_gat.shape[0]))
for i in range(load_array_gat.shape[0]):
    errs[i] = np.abs( np.max(load_array_gat[i]) - np.min(load_array_gat[i]) )
    means[i] = np.mean( load_array_gat[i] )
    stds[i] = np.std( load_array_gat[i] )
x = [1,2,3,4,5,6,7,8,9,10,11,12]

plt.figure()
plt.errorbar(x,means,yerr=stds,fmt='o',ecolor = 'cyan',color='black')
plt.xlabel('Instance')
plt.ylabel('Return')
plt.title('ARMA RETURNS')
#plt.savefig("ARMA_RETURNS_PLOT22_std.pdf")


#%% PLOT LSTM

#errs = np.zeros((load_array_lstm.shape[0]))
#means = np.zeros((load_array_lstm.shape[0]))
#stds = np.zeros((load_array_lstm.shape[0]))
#for i in range(load_array_lstm.shape[0]):
#    errs[i] = np.abs( np.max(load_array_lstm[i]) - np.min(load_array_lstm[i]) )
#    means[i] = np.mean( load_array_lstm[i] )
#    stds[i] = np.std( load_array_lstm[i] )
#
#x = [1,2,3,4,5,6,7,8,9,10,11,12]
#
#plt.figure()
#plt.errorbar(x,means,yerr=stds,fmt='o',ecolor = 'cyan',color='black')
#plt.xlabel('Instance')
#plt.ylabel('Return')
#plt.title('LSTM RETURNS')
##plt.savefig("LSTMN_RETURNS_PLOT2_std.pdf")



