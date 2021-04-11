# DSAI-HW2-2021
## Overview
The purpose of this repo is to determine the daily action and make the best profit for the future trading in the stock market.

## Data Collection
The `training_data.csv` and `testing_data.csv` are provided. Aside from the given **open-high-low-close**, I add additional one-hot encoded features to provide the information of **holding** or **shorting** stock and two-days-ahead prediction for more precise decision.

## Model
According to the instructions, the aim is to maximize the resulting accumulated profit via deciding the **Buy**, **NoAction**, and **Sell** actions based on the current **open-high-low-close** prices. Thus I turned to reinforcement learning (RL), which learns a policy through the resulting rewards. The deep q network (DQN) is a RL technique with Bellman equation as the update function and is suitable for continuous state space. Gated recurrent unit (GRU) is used in the DQN for modeling the q-values.

The accumulated profit over each episode are drawn below:

