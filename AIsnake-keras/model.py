import keras
from keras.models import Sequential, Model, load_model
from keras.layers import BatchNormalization, Flatten
from keras.layers import Input, Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from tensorflow.keras import initializers
import copy


class QTrainer_NN:
    def __init__(self, input_size, hidden_size, output_size, learn_rate, gamma, load_path):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = self.make_model(load_path)
        self.learn_rate = learn_rate
        self.gamma = gamma
        
        
    def make_model(self, load_path):
        if load_path != '':
            model = load_model(load_path)
        else:
            model = Sequential()
            model.add(Dense(256,activation='relu'))
            #model.add(Dense(128,activation='relu'))
            model.add(Dense(self.output_size))
            optimizer = Adam()
            model.compile(loss='mean_squared_error', optimizer=optimizer,
                               metrics=['accuracy'])
            
        return model

    def train_step(self, state, action, reward, next_state, done):
        
        # because train_short_memory() and train_long_memory() both call train_step(), so we need to know which one calls this function
        if (type(done) == bool):  # train_short_memory calls, so the types of some parameters need to be changed 
            done = np.array([done])    
            reward = np.array([reward])
            state = state.reshape((1, -1))
            next_state = next_state.reshape((1,-1))
        
        state = state.astype(dtype='float32')
        next_state = next_state.astype(dtype='float32')
        
        
        # predicted Q valuew with current state        
        target = self.model.predict(state)              
        # Q_new = r+ discount_rate*max(next_predicted Q value)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:  
                Q_new = reward[idx] + self.gamma*np.max(self.model.predict(next_state[idx].reshape((1,-1))))
            
            target[idx][np.argmax(action[idx])] = Q_new
                
        #print(targets.shape)
        loss = self.model.train_on_batch(state, target)
        #print(loss)
    
    