import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import numpy as np
import itertools



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
        self.n_stock, self.n_step = self.stock_price_history.shape

        # instance attributes
        self.init_invest = init_invest
        self.slide = slide
        self.model = model
        self.cur_step = None
        self.stock_owned = None
        self.stock_price = None
        self.cash_in_hand = None

        # action space
        self.action_space = spaces.Discrete(3 ** self.n_stock)

        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)
        stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]
        price_range = [[0, mx] for mx in stock_max_price]
        cash_in_hand_range = [[0, init_invest * 2]]
        # self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)
        # print('\nobservation space',self.observation_space)
        # self.observation_space = ()
        self.observation_space = 200 # stocks * slide
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

        # print(len(self.stock_price_history[:, self.cur_step]),self.stock_price_history[:, self.cur_step])

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
        if model in ['conv1d', 'lstm']:
            if self.cur_step < self.slide:
                for i in range(0,(self.slide-self.cur_step)):
                    obs.append(self.stock_price_history[:,0])
                for i in range(0,self.cur_step + 1):
                    obs.append(self.stock_price_history[:,i])
            else:
                for i in range((self.cur_step-self.slide), self.cur_step + 1):
                    obs.append(self.stock_price_history[:,i])
            obs = obs[-self.slide:]
            obs = np.array(obs)
            obs = np.reshape(obs, (1, obs.shape[0], obs.shape[1]))
            if model == 'lstm':
                return obs.T
            else:    
                return obs
        else:   # dnn 
            obs.extend(self.stock_owned)
            obs.extend(list(self.stock_price))
            obs.append(self.cash_in_hand)
            obs =  np.reshape(np.array(obs),(1,len(obs)))
            return obs 

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _trade(self, action):

        action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        action_vec = action_combo[action]

        for i, a in enumerate(action_vec): # i means index, a means action
            if a == 0: # sell
                # for j in range(i, 4 * i):
                for j in range(i, len(action_vec)):
                    if j < self.n_stock:
                        self.cash_in_hand += self.stock_price[j] * self.stock_owned[j]
                        self.stock_owned[j] = 0
                    else:
                        break
            elif a == 2: # buy
                # for j in range(i, 4 * i):
                for j in range(i, len(action_vec)):
                    if j < self.n_stock and self.cash_in_hand > self.stock_price[i] * 200:
                        self.stock_owned[j] += 200  # buy one share
                        self.cash_in_hand -= self.stock_price[j] * 200
                    else:
                        break


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
