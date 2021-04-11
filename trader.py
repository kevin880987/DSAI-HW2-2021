
import inspect
import matplotlib.pyplot as plt
from datetime import datetime

from keras.models import load_model

from utils import *
from stock_market import StockMarket
from dqn import DQN

# self=trader
class Trader():
    def __init__(self):
        self.verbose = True

    def train(self, training_data):
        episodes = 100 # 10 #
        epsilon = 1.0
        epsilon_min = 0.01
        # epsilon_decay = 0.90

        starttime = datetime.now()
        print()
        print()
        print(inspect.currentframe().f_code.co_name)
        print('\tstart time:', starttime)
        print()

        env = StockMarket(training_data) # data is only an information of prices in the environment
        self.agent = DQN(env)
        self.train_history = []
        for episode in range(episodes):
            if self.verbose:
                print('Episode:', episode)
                print()
                
            curr_state = env.reset()
            # epsilon *= epsilon_decay
            e = max(epsilon_min, epsilon*(1-episode/episodes))
            # if np.random.random()<epsilon:
            #     action = random.sample(list(filter(lambda x: x==x, available_action_space)), 1)[0]
            #     print(action)
            while True:
                action = self.agent.act(curr_state, e)
                reward, next_state, done = env.step(action)
                
                self.agent.remember(curr_state, action, reward, next_state, done)
                self.agent.replay()
                self.agent.target_train()

                curr_state = next_state

                if done:
                    break

            self.train_history.append(env.stock_trader.accumulated_profit)
            
            # Visualize single trial
            env.trial_history.plot()
            plt.title('Trial Profit')
            plt.xlabel('Episode')
            plt.ylabel('Profit')
            plt.show()

            # Visualize training status
            pd.DataFrame(self.train_history, columns=['Accumulated Profit']).plot()
            plt.title('Training Accumulated Profit')
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.show()
            
            if self.verbose:
                print('Final Accumulated Profit:\t', env.stock_trader.accumulated_profit)
                print('='*50)
                print()
        
        # Save model
        self.agent.save_model('model.h5')
        
        endtime = datetime.now()
        print()
        print(inspect.currentframe().f_code.co_name)
        print('\tend time:', endtime)
        print('\ttime consumption:', endtime-starttime)
        print()
        print()
                
    def predict_action(self, row):
        try:
            _, curr_state, _ = self.predict_env.step(self.prev_action, row)
        except:
            self.predict_env = StockMarket()
            curr_state = self.predict_env.reset(row)

        try:
            if self.agent==self.agent:
                pass
        except:
            env = StockMarket() # data is only an information of prices in the environment
            self.agent = DQN(env)
            self.agent.model = load_model('model.h5')

        action = self.agent.act(curr_state)
        
        self.prev_action = action
        return action
        
    # def re_training(self):
        
        

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    training_data = load_data(args.training)
    trader = Trader()
    trader.train(training_data)
    
    testing_data = load_data(args.testing)
    with open(args.output, 'w') as output_file:
        for row in testing_data.values:
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(row)
            output_file.write(f'{str(action)}\n')

            # this is your option, you can leave it empty.
            # trader.re_training(i)

