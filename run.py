import pickle
import time
import numpy as np
import argparse
import re
import logging
import tqdm
import pickle

from datetime import datetime
from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir, plot_all

# stock_name = "tech"
# stock_table = "tech_table"


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
    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')

    slide = 20
    # if args.model_type in ['conv1d','lstm']:
    #     window = True
    # else:
    #     window = False
        
    stock_name = args.stock
    # stock_table = f"{stock_name}_table"
    stock_table = f"{stock_name.split('_')[0]}_table"
    timestamp = time.strftime('%Y%m%d%H%M')

    # data = get_data(stock_name, stock_table, window, slide)
    data = get_data(stock_name, stock_table)

    # configure logging
    logging.basicConfig(
        filename=f'logs/{args.mode}_{args.model_type}_{stock_name}_{timestamp}.log', 
        filemode='w',
        format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S', 
        level=logging.INFO
        )
    logging.info(f'Mode:                     {args.mode}')
    logging.info(f'Model Type:               {args.model_type}')
    logging.info(f'Training Object:          {stock_name} portfolio')
    # logging.info(f'Test Period:              {test_date[0]} ~ {test_date[-1]}, {len(train_date)} days')
    logging.info(f'Model Weights:            {args.weights}')
    logging.info(f'Training Episode:         {args.episode}')
    logging.info(f'Initial Invest Value:    ${args.initial_invest:,}')
    logging.info(f'='*30)

    # env = TradingEnv(train_data, args.initial_invest)
    env = TradingEnv(data, args.model_type, args.initial_invest, slide)
    # state_size = env.observation_space
    action_size = env.action_space.n
    state_size = np.array(env.reset()).shape
    print('state_size',state_size)
    agent = DQNAgent(state_size, action_size, args.mode, args.model_type)
    scaler = get_scaler(env)

    portfolio_value = []

    logging.info(f'{args.mode} start')
    if args.mode == 'test':
        # remake the env with test data
        # env = TradingEnv(test_data, args.initial_invest)
        env = TradingEnv(data, args.model_type, args.initial_invest,slide)
        # load trained weights
        agent.load(f'weights/{args.weights}.h5')
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
        for time in range(env.n_step):
            action = agent.act(state)
            action_list.append(action)
            next_state, reward, done, info = env.step(action)
            # next_state = scaler.transform([next_state])
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            if args.mode == "test":
                daily_portfolio_value.append(info['cur_val'])
            state = next_state
            if done:
                if args.mode == "test":
                    # print(daily_portfolio_value)
                    plot_all(stock_name, daily_portfolio_value, env)
                print(len(action_list),action_list)    
                daily_portfolio_value = []
                # logging.info("episode: {}/{}, episode end value: {}".format(e + 1, args.episode, info['cur_val']))
                logging.info(f"episode: {e+1}/{args.episode}, episode end value: {info['cur_val']}")
                portfolio_value.append(info['cur_val']) # append episode end portfolio value
                break
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay(args.batch_size)
        if args.mode == 'train' and (e+1) % 10 == 0:  # checkpoint weights
            agent.save(f'weights/{stock_name}_{args.model_type}-{timestamp}-ep{e+1}.h5')
    if args.mode == 'train':
        print("mean portfolio_val:", np.mean(portfolio_value))
        print("median portfolio_val:", np.median(portfolio_value))
    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)

    logging.info(f'{args.mode} ending')
