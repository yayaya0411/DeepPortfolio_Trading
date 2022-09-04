import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import numpy as np
import itertools
import pickle
import os
from config import scaler_file, buy_stock, to_ratio, to_gray

import warnings
warnings.filterwarnings("ignore")


class TradingEnv(gym.Env):
    """
  A 3-stock (MSFT, IBM, QCOM) trading environment.

  State: [# of stock owned, current stock prices, cash in hand]
    - array of length n_stock * 2 + 1
    - price is discretized (to integer) to reduce state space
    - use close price for each stock
    - cash in hand is evaluated at each step based on action performed

  Action: sell (0), hold (1), and buy (2)
    - when selling, sell all the shares
    - when buying, buy as many as cash in hand allows
    - if buying multiple stock, equally distribute cash in hand and then utilize the balance
  """

    def __init__(self, train_data, model, init_invest=20000, slide = 20):
        # data
        # self.n_industry = 5
        # self.buy_date = [[] for _ in range(self.n_industry)]
        # self.sell_date = [[] for _ in range(self.n_industry)]
        self.stock_price_history = train_data # round up to integer to reduce state space
        self.stock_price_ratio = pd.DataFrame(train_data).pct_change().iloc[1:].values
        self.stock_price_gray = pd.DataFrame(self.stock_price_ratio).apply(lambda x : np.where(x>=0,1,0)).values
        self.n_stock, self.n_step = self.stock_price_history.shape

        # instance attributes
        self.init_invest = init_invest
        self.slide = slide
        self.model = model
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None
        self.buy_stock = buy_stock
        self.scaler = pickle.load(open(os.path.join('scaler',scaler_file), 'rb'))
        # action space
        self.action_space = spaces.Discrete(3 ** self.n_stock)

        # observation space: give estimates in order to sample and build scaler
        # stock_max_price = self.stock_price_history.max(axis=1)
        # stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
        # price_range = [[0, mx] for mx in stock_max_price]
        # cash_in_hand_range = [[0, init_invest * 2]]
        # self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)
        # self.observation_space = ()
        # self.observation_space = 200 # stocks * slide
        # seed and start
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.cur_step = 0
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]


        self.cash_in_hand = self.init_invest
        return self._get_obs(self.model)

    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.cur_step += 1
        self.stock_price = self.stock_price_history[:, self.cur_step]  # update price
        self._trade(action)
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}
        return self._get_obs(self.model), reward, done, info

    def _get_obs(self, model):
        obs = []
        stock = self.stock_price_history
        assert to_ratio != to_gray, f'\ncheck config.py to_ratio and to_gray both True\n'
        
        if to_ratio:
            stock = self.stock_price_ratio 
        if to_gray:
            stock = self.stock_price_gray
               
        if model in ['conv1d', 'conv2d', 'lstm','transformer']:
            if self.cur_step < self.slide:
                for i in range(0,(self.slide-self.cur_step)):
                    obs.append(stock[:,0])
                for i in range(0,self.cur_step + 1):
                    obs.append(stock[:,i])
            else:
                for i in range((self.cur_step-self.slide), self.cur_step + 1):
                    obs.append(stock[:,i])
            obs = obs[-self.slide:]
            obs = np.array(obs)
            if not to_ratio:
                obs = self.scaler.transform(obs)
            obs = np.reshape(obs, (1, obs.shape[0], obs.shape[1]))
            # obs = self.scaler.transform(obs)
            if model in ['lstm','transformer']:
                return obs.T
            else:    
                return obs
        else:   # dnn 
            # obs.extend(self.stock_owned)
            # print(self.stock_price)
            obs.extend(list(self.stock_price))
            # obs.append(self.cash_in_hand)
            # obs = np.reshape(obs,(1,obs.shape[1]))
            # print(obs)
            obs = np.reshape(obs,(1,len(obs)))
            if not to_ratio:
                obs = self.scaler.transform(obs)
            # print(obs)
            return obs 

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _trade(self, action):

        action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        action_vec = action_combo[action]
        # print('='*40+' trading '+'='*40)
        for i, a in enumerate(action_vec): # i means index, a means action
            # print(f'trading start index {i} action {a}')
            if a == 0: # sell
                if i < self.n_stock:
                    self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                    self.stock_owned[i] = 0
                    # print(f'{action_vec} sell index {i}, action {a}, n stock {self.n_stock}, stock_owned {self.stock_owned}')
                    # print('stock price\n',self.stock_price,'\nstock_owned\n',self.stock_owned,'\n')
                    # print(f'trading stock price {self.stock_price[i]}, now stock owned {self.stock_owned[i]},\n')
                else:
                    # print(f'{action_vec} sell break index {i}, action {a} ',self.n_stock, self.stock_owned)
                    break
            elif a == 1: #hold
                # print('hold')    
                pass
            
            elif a == 2: # buy
                if (i < self.n_stock) and (self.cash_in_hand > (self.stock_price[i] * self.buy_stock)):
                    self.stock_owned[i] += self.buy_stock  # buy 1000 share
                    self.cash_in_hand -= self.stock_price[i] * self.buy_stock
                    # print(f'{action_vec} buy index {i}, action {a}, {self.n_stock}, cash ${self.cash_in_hand}, cost ${(self.stock_price[i] * self.buy_stock)}\nstock owned {self.stock_owned}')    
                    # print(self.stock_price,'\n',self.stock_owned,'\n')
                    # print(self.stock_price[i],'\n',self.stock_owned[i],'\n')
                else:
                    # print(f'{action_vec} buy break index {i}, action {a} ',self.n_stock,self.cash_in_hand, (self.stock_price[i] * self.buy_stock), self.stock_owned)
                    break

        # ================original==================
        # for i, a in enumerate(action_vec): # i means index, a means action
        #     print(f'trading start index {i} action {a}')
        #     if a == 0: # sell
        #         # for j in range(i, 4 * i):
        #         for j in range(i, len(action_vec)):
        #             if j < self.n_stock:
        #                 self.cash_in_hand += self.stock_price[j] * self.stock_owned[j]
        #                 self.stock_owned[j] = 0
        #                 print(f'{action_vec} sell index {i}, action {a}, stock {j}', self.n_stock, self.stock_owned)    
        #             else:
        #                 print(f'{action_vec} sell break index {i}, action {a}, stock {j}',self.n_stock, self.stock_owned)
        #                 break
        #     elif a == 2: # buy
        #         # for j in range(i, 4 * i):
        #         for j in range(i, len(action_vec)):
        #             if (j < self.n_stock) and (self.cash_in_hand > (self.stock_price[i] * self.buy_stock)):
        #                 self.stock_owned[j] += self.buy_stock  # buy 1000 share
        #                 self.cash_in_hand -= self.stock_price[j] * self.buy_stock
        #                 print(f'{action_vec} buy index {i}, action {a}, stock {j}',self.n_stock,self.cash_in_hand, (self.stock_price[i] * self.buy_stock), self.stock_owned)    
        #             else:
        #                 print(f'{action_vec} buy break index {i}, action {a}, stock {j}',self.n_stock,self.cash_in_hand, (self.stock_price[i] * self.buy_stock), self.stock_owned)
        #                 break
        # ================original==================


        # all combo to sell(0), hold(1), or buy(2) stocks
        # action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        # action_vec = action_combo[action]

        # # one pass to get sell/buy index
        # sell_index = []
        # buy_index = []
        # for i, a in enumerate(action_vec):
        #     if a == 0:
        #         sell_index.append(i)
        #     elif a == 2:
        #         buy_index.append(i)
        # # two passes: sell first, then buy; might be naive in real-world settings
        # if sell_index:
        #     for i in sell_index:
        #         self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        #         self.stock_owned[i] = 0
        # if buy_index:
        #     can_buy = True
        #     while can_buy:
        #         for i in buy_index:
        #             if self.cash_in_hand > self.stock_price[i]:
        #                 self.stock_owned[i] += 100 # buy one share
        #                 self.cash_in_hand -= self.stock_price[i]
        #             else:
        #                 can_buy = False    
