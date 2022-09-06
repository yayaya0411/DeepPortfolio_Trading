import pickle
import time
import numpy as np
import argparse
import re
import logging
import tqdm
import pickle
import pandas as pd

from datetime import datetime
from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, maybe_make_dir, plot_all 
# stock_name = "tech"
# stock_table = "tech_table"
from config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=50, help='number of episode to run')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size for experience replay')
    parser.add_argument('-i', '--initial_invest', type=int, default=1000000, help='initial investment amount')
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    parser.add_argument('-t', '--model_type', type=str, required=True, help='"dnn", "conv1d" or "lstm"')
    parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
    parser.add_argument('-s', '--stock', type=str, required=True, default='tech', help='stock portfolios')
    args = parser.parse_args()

    maybe_make_dir('logs')
    maybe_make_dir(f'logs/{args.mode}/{args.model_type}')
    maybe_make_dir('weights')
    maybe_make_dir(f'weights/{args.model_type}')
    maybe_make_dir('portfolio_val')
    maybe_make_dir('scaler')

    # slide = 10
            
    stock_name = args.stock
    # stock_table = f"{stock_name}_table"
    stock_table = f"{stock_name.split('_')[0]}_table"
    print(stock_table)
    timestamp = time.strftime('%Y%m%d%H%M')
    datestamp = time.strftime('%Y%m%d')

    data = get_data(stock_name, stock_table)

    stock_code = pd.read_csv('data/{}.csv'.format(stock_name)).drop(columns="DateTime")
    stock_code = sorted(stock_code.columns)

    env = TradingEnv(data, args.model_type, args.initial_invest, slide)
    action_size = env.action_space.n
    state_size = np.array(env.reset()).shape
    
    model_prefix = f'{timestamp}_{stock_name}_{args.model_type}'
    agent = DQNAgent(state_size, action_size, args.mode, args.model_type, model_prefix)


    # configure logging
    logging.basicConfig(
        filename=f'logs/{args.mode}/{args.model_type}/{model_prefix}_{args.mode}.log', 
        filemode='w',
        format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S', 
        level=logging.INFO
        )
    logging.info(f'Mode:                     {args.mode}')
    logging.info(f'Model Type:               {args.model_type}')
    logging.info(f'Training Object:          {stock_name}')
    logging.info(f'Portfolio Stock:          {stock_code}')
    logging.info(f'Window Slide:             {slide} days')
    logging.info(f'Turn to Ratio:            {to_ratio}')
    logging.info(f'Turn to Gray:             {to_gray}')
    logging.info(f'Buy/Sell Stocks:          {env.buy_stock} per action')
    logging.info(f'Model Weights:            {args.weights}')
    logging.info(f'Training Episode:         {args.episode}')
    logging.info(f'Initial Invest Value:    ${args.initial_invest:,}')
    logging.info(f'='*30)

    # env = TradingEnv(train_data, args.initial_invest)
    # state_size = env.observation_space
    # scaler = get_scaler(env)

    portfolio_value = []

    logging.info(f'{args.mode} start')
    if args.mode == 'test':
        # remake the env with test data
        # env = TradingEnv(test_data, args.initial_invest)
        env = TradingEnv(data, args.model_type, args.initial_invest, slide)
        # load trained weights
        agent.load(f'weights/{args.model_type}/{args.weights}.h5')
        # when test, the timestamp is same as time when weights was trained
        # timestamp = stock_name + '_' +re.findall(r'\d{12}', args.weights)[0]
        # daily_portfolio_value = [env.init_invest]
        daily_portfolio_value = []
        
        args.episode = 1

    for e in tqdm.tqdm(range(args.episode)):
        state = env.reset()
        # state = scaler.transform([state])
        # state = scaler.transform(state)
        action_list=[]
        action_target = None
        action_target_cnts = 1
        reward_discount = 0.95
        for time in range(env.n_step):
            action = agent.act(state)
            action_list.append(action)
            next_state, reward, done, info = env.step(action)
                
            if args.mode == 'train':
                # encourage change policy
                if action == action_target:
                    reward *= reward * reward_discount ** action_target_cnts
                    action_target_cnts +=1
                else:
                    print(f'encourage avoid same action')
                    action_target_cnts = 1
                # remember and train    
                agent.remember(state, action, reward, next_state, done)
            if args.mode == "test":
                daily_portfolio_value.append(info['cur_val'])
            state = next_state
            if done:
                if args.mode == "test":
                    plot_all(stock_name, args.model_type, daily_portfolio_value, env)
                print(len(action_list),action_list)    
                daily_portfolio_value = []
                logging.info(f"episode: {e+1}/{args.episode}, episode end value: {info['cur_val']}")
                portfolio_value.append(info['cur_val']) # append episode end portfolio value
                break
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)
        if args.mode == 'train' and (e+1) % 10 == 0:  # checkpoint weights
            agent.save(f'weights/{args.model_type}/{model_prefix}-ep{str(e+1).zfill(3)}.h5')
    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)

    logging.info(f'{args.mode} ending')
