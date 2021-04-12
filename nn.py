# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:56:59 2021

@author: POLAB
"""

import numpy as np
import matplotlib.pyplot as plt
import inspect
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

class NN():
    # def __init__(self):
    #     pass
    
    def __init__(self, input_shape, output_size, layers=3, units=3, 
                 dropout=0.05, loss='mse', optimizer='adam'):
        self.verbose = False
        self.display = True
        
        model = Sequential()
        for i, u in zip(range(layers), np.linspace(units, 1, layers)):
            if i<layers-1:
                return_sequences = True
            else:
                return_sequences = False
                
            model.add(GRU(units=int(np.round(max(1, u))), input_shape=input_shape, 
                             return_sequences=return_sequences))
            model.add(Dropout(dropout))
        
        model.add(Dense(output_size)) # model.add(TimeDistributed(Dense(1)))
        model.compile(loss=loss, optimizer=optimizer)
        
        self.model = model

# self=predict_model
    def train(self, X, Y, epochs=24, model=None):
        # Both X and Y are 2d array

        starttime = datetime.now()
        print()
        print()
        print(inspect.currentframe().f_code.co_name)
        print('\tstart time:', starttime)
        print()

        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Normalize Y
        self.Y_scaler = StandardScaler().fit(Y)
        Y = self.Y_scaler.transform(Y)

        if model is None:
            fit_history = self.model.fit(X, Y, epochs=epochs, verbose=self.verbose)
    
            if self.display:
                # Visualize
                plt.plot(fit_history.history['loss'])
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss Function Value')
                plt.legend(['Loss'])
                plt.show()
            
        else:
            self.model = model
        
        if self.display:
            # Visualize prediction
            Y_pred = self.predict(X)
            plt.plot(X[:, 0])
            plt.legend(['True'])
            for i, y in enumerate(Y_pred):
                plt.plot(i+np.arange(0, y.shape[0]), y)
            plt.title('Prediction')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.show()
            
            print()

        endtime = datetime.now()
        print()
        print(inspect.currentframe().f_code.co_name)
        print('\tend time:', endtime)
        print('\ttime consumption:', endtime-starttime)
        print()
        print()
        
    def predict(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y_pred = self.model.predict(x)
        y_pred = self.Y_scaler.inverse_transform(y_pred)
        
        return y_pred
        
    def save_model(self, fn):
        self.model.save(fn)
    