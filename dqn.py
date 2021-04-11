# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 00:28:54 2021

@author: kevin
"""

import numpy as np
import random

from collections import deque
from numpy.ma import masked_array

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import Adam

# self=DQN(env)
class DQN():
    def __init__(self, env):      
        self.env = env
        
        self.gamma = 0.80
        # self.learning_rate = 0.005
        self.tau = .125
        self.verbose = True
    
        self.memory = deque(maxlen=1000)
        
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        layers = 4 # 8 # 
        units = int(round(self.env.state_size/2)) # 3 # 4 # 
        dropout = 0.05

        loss = 'mse'
        optimizer = 'adam'

        model = Sequential()
        for i, u in zip(range(layers), np.linspace(units, 1, layers)):
            if i<layers-1:
                return_sequences = True
            else:
                return_sequences = False
                
            model.add(GRU(units=int(np.round(max(1, u))), input_shape=(self.env.state_size, 1), 
                             return_sequences=return_sequences))
            model.add(Dropout(dropout))
        
        model.add(Dense(len(self.env.action_space))) # model.add(TimeDistributed(Dense(1)))
        model.compile(loss=loss, optimizer=optimizer)
        # model.summary()
        
        return model

    def act(self, state, epsilon=None):
        mask = self.env.get_available_actions(state)
        available_action_space = masked_array(self.env.action_space, mask)

        if epsilon is not None and np.random.random()<epsilon:
            action = random.sample(list(filter(lambda x: x==x, available_action_space)), 1)[0]
        else:
            prediction = masked_array(self.model.predict(state)[0], mask)
            action = available_action_space[np.argmax(prediction)]
            
        return action
    
    def remember(self, state, action, reward, next_state, done):
        if self.verbose:
            print('State:\t', state.flatten())
            print('Prediction:\t', self.model.predict(state)[0])
            print('Action Mask:\t', self.env.get_available_actions(state))
            print('Action:\t', action)
            print('Reward:\t', reward)
            print('Next State:\t', next_state.flatten())
            print()

        self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        batch_size = 64
        epochs = 5

        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][self.env.action_space.index(action)] = reward
            else:
                mask = self.env.get_available_actions(state)
                Q_future = masked_array(self.target_model.predict(next_state)[0], mask).max()
                target[0][self.env.action_space.index(action)] \
                    = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=epochs, verbose=0)
                        
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)
        
        
        