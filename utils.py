# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:42:33 2021

@author: kevin
"""

import pandas as pd
import numpy as np
import os

def load_data(file_name):
    root_fp = os.getcwd() + os.sep
    df = pd.read_csv(root_fp+file_name, header=None)

    return df

from enum import IntEnum
class Action(IntEnum):
    BUY = 1
    NO_ACTION = 0
    SELL = -1
    
class InvalidActionError(Exception):
    pass

class StockNumExceedError(Exception):
    pass

class InvalidActionNumError(Exception):
    pass

def shift(arr, n: int):
    # arr is a 2d array
    
    new_arr = np.full_like(arr, np.nan)
    
    if n==0:
        new_arr = arr
    
    elif n>0:
        new_arr[n: ] = arr[: -n]
    
    elif n<0:
        n = -n
        new_arr[: -n] = arr[n: ]
    
    return new_arr