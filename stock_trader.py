# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 01:02:31 2021

@author: kevin
"""

from utils import *

class StockTrader:
    def __init__(self):
        self.accumulated_profit = 0
        self.holding_price = None
        self.short_selling_price = None
        self.M = -999999999

    @property
    def is_holding_stock(self):
        return self.holding_price is not None

    @property
    def is_shorting_stock(self):
        return self.short_selling_price is not None

    def perform_action(self, action_code: int, stock_price: float):
        if action_code == Action.BUY:
            profit = self.buy(stock_price)
        elif action_code == Action.SELL:
            profit = self.sell(stock_price)
        elif action_code == Action.NO_ACTION:
            profit = 0
        else:
            raise InvalidActionError('Invalid Action')
        
        return profit
    
    def buy(self, stock_price: float):
        if self.is_holding_stock:
            profit = self.M
            # raise StockNumExceedError('You cannot buy stocks when you hold one')
        elif self.is_shorting_stock:
            profit = self.short_selling_price - stock_price
            self.accumulated_profit += profit
            self.short_selling_price = None
        else:
            profit = 0
            self.holding_price = stock_price

        return profit

    def sell(self, stock_price: float):
        if self.is_shorting_stock:
            profit = self.M
            # raise StockNumExceedError("You cannot sell short stocks when you've already sell short one")
        elif self.is_holding_stock:
            profit = stock_price - self.holding_price
            self.accumulated_profit += profit
            self.holding_price = None
        else:
            profit = 0
            self.short_selling_price = stock_price

        return profit
            
        
        