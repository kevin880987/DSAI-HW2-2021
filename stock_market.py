# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 00:44:05 2021

@author: kevin
"""

import numpy as np
import pandas as pd

from stock_trader import StockTrader
from utils import Action

# data=training_data
# data=testing_data
# self=StockMarket(data)
# self=env
class StockMarket():
    def __init__(self, data=None):
        if data is not None:
            self.data = data.reindex()
        
        self.state_size = 6
        self.action_space = [1, 0, -1]

        self.trial_length = 20
        self.open_index = 0
        
    def build_state(self, prices):
        
        open_price = prices[self.open_index]

        # Quote change
        prices /= open_price
        
        # Inventory
        inventory = [self.stock_trader.is_holding_stock, 
                     self.stock_trader.is_shorting_stock] 
        inventory = [1 if x else 0 for x in inventory]
        self.is_holding_stock_index = 4
        self.is_shorting_stock_index = 5

        inventory_price = [self.stock_trader.holding_price, 
                           self.stock_trader.short_selling_price]
        inventory_price = list(filter(lambda x: x is not None, inventory_price))
        if len(inventory_price)==1:
            prices[self.open_index] = inventory_price[0] / open_price
        
        # inventory = np.array([x-open_price if x is not None else 0 for x in inventory])
        # inventory -= open_price
        # inventory = np.array([30, None])
        
        # Build state
        state = np.concatenate((prices, inventory))
        state = state.reshape(1, self.state_size, 1)
        
        return state
        
    def reset(self, curr_prices=None):
        self.step_ctr = 0
        self.done = False
        
        self.stock_trader = StockTrader()
        self.trial_history = []

        if curr_prices is None:
            curr_prices = self.data.iloc[: -self.trial_length+1].sample()
            self.curr_index = curr_prices.index[0]
            curr_prices = curr_prices.values.flatten()

        curr_state = self.build_state(curr_prices)
        return curr_state
        
    def step(self, action, next_prices=None):
        self.step_ctr += 1
        if next_prices is None:
            next_index = self.curr_index+1
            self.curr_index = next_index
            next_prices = self.data.loc[[next_index]].values.flatten()
        
        reward = self.stock_trader.perform_action(action, next_prices[self.open_index])
        next_state = self.build_state(next_prices)
        
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
        