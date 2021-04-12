# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 00:44:05 2021

@author: kevin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model

from stock_trader import StockTrader
from utils import Action, shift
from nn import NN

# data=training_data
# data=testing_data
# self=StockMarket(training_data)
# self=env
class StockMarket():
    def __init__(self, data):
        # data is a 2d array
        
        self.data = data
        
        self.state_size = 9
        self.action_space = [1, 0, -1]
        self.trial_length:int = 20
        self.open_index:int = 0 # numerical index

        # Predict
        self.past_step = 5
        self.future_step = 3
        self.epochs = 100 # 2 # 50 # 600 # 
        
        
    def create_predict_model(self):
        X = np.empty((self.data.shape[0], 0))
        for n in range(self.past_step):
            X = np.hstack((X, shift(self.data, n)))
        Y = np.empty((self.data.shape[0], 0))
        for n in range(1, self.future_step+1):
            Y = np.hstack((Y, shift(self.data[:, [self.open_index]], -n)))
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
        X = X[mask]
        Y = Y[mask]
        
        predict_model = NN(input_shape=(X.shape[1], 1), 
                           output_size=Y.shape[1], 
                           layers=6, units=max(2, int(round(X.shape[1]/2))))

        predict_model.train(X, Y, epochs=self.epochs) # , model=load_model('predict_model.h5')) # 

        self.predict_model = predict_model
        predict_model.save_model('predict_model.h5')

    def build_state(self):
        # prices is a 1d array
        
        prices = self.data[self.curr_index]
        open_price = prices[self.open_index]

        # Quote change
        volatility = prices / open_price
        
        inventory_price = [self.stock_trader.holding_price, 
                           self.stock_trader.short_selling_price]
        inventory_price = list(filter(lambda x: x is not None, inventory_price))
        if len(inventory_price)==1:
            volatility[self.open_index] = inventory_price[0] - open_price
        
        # Inventory
        inventory = [self.stock_trader.is_holding_stock, 
                     self.stock_trader.is_shorting_stock] 
        inventory = [1 if x else 0 for x in inventory]
        self.is_holding_stock_index = 4
        self.is_shorting_stock_index = 5

        # inventory = np.array([x-open_price if x is not None else 0 for x in inventory])
        # inventory -= open_price
        # inventory = np.array([30, None])
        
        # Prediction
        x = self.data[self.curr_index: self.curr_index-self.past_step: -1]
        x = x.reshape((1, x.size, 1))
        prediction = self.predict_model.predict(x)[0]
        prediction /= open_price

        # Build state
        state = np.concatenate((volatility, inventory, prediction))
        if state.size != self.state_size:
            raise ValueError('Incompatible state size.')
        state = state.reshape(1, state.size, 1)
        
        return state
        
    def reset(self, curr_prices=None):
        self.step_ctr = 0
        self.done = False
        
        self.stock_trader = StockTrader()
        self.trial_history = []

        if curr_prices is None: # randomly select a start point from self.data
            self.curr_index = int(round(np.random.rand() * (self.data.shape[0]-self.trial_length)))
            
        else: # add the passed curr_prices to the self.data and reset
            self.data = np.vstack((self.data, curr_prices))
            self.curr_index = self.data.shape[0] - 1
            
        curr_state = self.build_state()
        return curr_state
        
    def step(self, action, next_prices=None):
        self.step_ctr += 1
        
        self.curr_index += 1
        if next_prices is None: # continue
            pass
        else:
            self.data = np.vstack((self.data, next_prices))
                    
        reward = self.stock_trader.perform_action(action, self.data[self.curr_index, self.open_index])
        next_state = self.build_state()
        
        self.trial_history.append([self.stock_trader.accumulated_profit, reward])
        
        if self.step_ctr>=self.trial_length-1:
            self.done = True
            self.trial_history \
                = pd.DataFrame(self.trial_history, 
                               columns=['Accumulated Profit', 'Step Profit'])
            
        return reward, next_state, self.done

    def get_available_actions(self, state):
        state = state.flatten()
        mask = []
        for a in self.action_space:
            if a==Action.BUY:
                m = 1 if state[self.is_holding_stock_index]!=0 else 0
            elif a==Action.SELL:
                m = 1 if state[self.is_shorting_stock_index]!=0 else 0
            elif a==Action.NO_ACTION:
                m = 0
            else:
                m = 1
            mask.append(m)
            
        return mask
        