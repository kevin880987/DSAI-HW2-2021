# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 00:28:54 2021

@author: kevin
"""

import numpy as np
import random

from collections import deque
from numpy.ma import masked_array
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import Adam

from nn import NN

# self=self.agent
# self=DQN(env)
class DQN():
    def __init__(self, env):      
        self.env = env
        
        self.gamma = 0.95
        # self.learning_rate = 0.005
        self.tau = .125
        self.verbose = True
    
        self.memory = deque(maxlen=1000)
        
        self.state_size = env.state_size

        self.model = self.create_model()
        self.target_model = self.create_model()
                
    def create_model(self):
        model = NN(input_shape=(self.state_size, 1), 
                   output_size=len(self.env.action_space), 
                   layers=6, units=int(round(self.state_size/2))).model
                
        return model

    def act(self, state, epsilon=None):
        state = state.reshape(state.shape[0], state.shape[1], 1)
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
        batch_size = 32
        epochs = 50

        if len(self.memory) < batch_size: 
            return

        states = np.empty((0, self.state_size, 1))
        targets = np.empty((0, len(self.env.action_space)))
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
                    
            states = np.vstack((states, state))
            targets = np.vstack((targets, target))
        
        fit_history = self.model.fit(states, targets, epochs=epochs, verbose=0)
        # plt.plot(fit_history.history['loss'])
        # plt.show()
                        
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self):
        self.model.save('model.h5')
        self.target_model.save('target_model.h5')
        
        
        