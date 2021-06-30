import os
import numpy as np
import random as ra
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, \
                        Flatten, Dense, Dropout, Reshape, Lambda, \
                        Concatenate, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras import regularizers
from keras import optimizers as op
import deeprec.metrics as mt

def build_deeprec_model(params, seq_len=10, random_state=None):
    """ """
    if random_state != None:
        os.environ['PYTHONHASHSEED']=str(random_state)
        np.random.seed(random_state)
        ra.seed(random_state)
#    keras.backend.clear_session()
    
    # build convolutional layers
    patch_size = params.hbond_major['filter_len']

    # construct the 1D input layer
    channel, height, length = 4, 7, (2*seq_len+patch_size)
    input_data = Input(shape=(channel*height*length,), name="hbond_x")

    # construct the conv2D layer
    input_reshape = Reshape((channel,height,length))(input_data)
    input_split = Lambda(lambda x:tf.split(x, [4,3], axis=2))(input_reshape)
    input_major = input_split[0]
    input_minor = input_split[1]

    # conv2D for major
    conv_major = Conv2D(filters=params.hbond_major['nb_filter'], 
                        kernel_size=(params.hbond_major['filter_hei'], 
                                     params.hbond_major['filter_len']),
                        activation=params.hbond_major['activation'],
                        kernel_initializer='glorot_uniform',                                                                       
                        padding='valid',
                        data_format='channels_first',
                        kernel_regularizer=regularizers.l1_l2(
                                l1=params.hbond_major['l1'], 
                                l2=params.hbond_major['l2']))(input_major)
    pool_major = MaxPooling2D(pool_size=(params.hbond_major['pool_hei'],
                                params.hbond_major['pool_len']))(conv_major)
    flat_major = Flatten()(pool_major)
    
    # hidden layber for major
    if params.hbond_major['nb_hidden']==0:
        hidden_major = flat_major
    else:
        hidden_major = Dense(units=params.hbond_major['nb_hidden'],
                        activation=params.hbond_major['activation'],
                        kernel_regularizer=regularizers.l2(0),
                        activity_regularizer=regularizers.l1(0))(flat_major)        
        
    # conv2D for minor
    conv_minor = Conv2D(filters=params.hbond_minor['nb_filter'], 
                        kernel_size=(params.hbond_minor['filter_hei'], 
                                     params.hbond_minor['filter_len']),
                        activation=params.hbond_minor['activation'],
                        kernel_initializer='glorot_uniform',                                                                       
                        padding='valid',
                        data_format='channels_first',
                        kernel_regularizer=regularizers.l1_l2(
                                l1=params.hbond_minor['l1'], 
                                l2=params.hbond_minor['l2']))(input_minor)
    pool_minor = MaxPooling2D(pool_size=(params.hbond_minor['pool_hei'],
                                params.hbond_minor['pool_len']))(conv_minor)
    flat_minor = Flatten()(pool_minor)
    
    # hidden layber for minor
    if params.hbond_minor['nb_hidden']==0:
        hidden_minor = flat_minor
    else:        
        hidden_minor = Dense(units=params.hbond_minor['nb_hidden'],
                        activation=params.hbond_minor['activation'],
                        kernel_regularizer=regularizers.l2(0),
                        activity_regularizer=regularizers.l1(0))(flat_minor)
    
    # Construct the hidden layer
    merge = concatenate([hidden_major, hidden_minor])
    hidden = Dense(units=params.joint['nb_hidden'], 
                   activation=params.joint['activation'], 
                   kernel_regularizer=regularizers.l1_l2(
                           l1=params.joint['l1'], 
                           l2=params.joint['l2']))(merge)    
    dropout = Dropout(params.joint['drop_out'], noise_shape=None)(hidden)
    output = Dense(1, activation=params.target['activation'])(dropout)

    # summarize the model
    model = Model(inputs=input_data, outputs=output)
#    model.summary()
    model.compile(optimizer=op.Adam(lr=params.optimizer_params['lr']), 
                  loss=params.loss, metrics=[mt.r_squared])
    
    
    return model
    