import pickle
from datetime import datetime
import os
from matplotlib import pyplot as plt
import h5py
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdate
import matplotlib.dates as mdates


# create training window
def training_windows(df,time_slide):
    X=[]
    for i in range(time_slide, len(df)): 
        tmp = df[i-time_slide:i]
        # tmp = normalization(tmp)
        X.append(tmp)
    return X    

def get_data(stock_name, stock_tabel):
    """ Returns a 3 x n_step array """
    industry = pd.read_csv('data/{}.csv'.format(stock_tabel))["code"].astype("str")
    data = pd.read_csv('data/{}.csv'.format(stock_name)).drop(columns="DateTime")
    data = data[industry].astype("float")
    data = data.T
    # if window:
    #     X = training_windows(data, slide)
    # else:
    X = np.array(data)
    return X



# def get_scaler(env):
#     """ Takes a env and returns a scaler for its observation space """
#     low = [0] * (env.n_stock * 2 + 1)
#     high = []
#     max_price = env.stock_price_history.max(axis=1)
#     min_price = env.stock_price_history.min(axis=1)
#     max_cash = env.init_invest * 3  # 3 is a magic number...
#     max_stock_owned = max_cash // min_price
#     for i in max_stock_owned:
#         high.append(i)
#     for i in max_price:
#         high.append(i)
#     high.append(max_cash)

#     scaler = StandardScaler()
#     scaler.fit([low, high])
#     return scaler


def input_build(new_x, y, ori_y):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    # val_x = []
    # val_y = []
    
    tr_va = int(len(new_x)*8/10)
    y = list(y)

    # original add all data into training
    train_x = train_x + new_x[:tr_va]
    train_y = train_y + y[:tr_va]
    # val_x = val_x + new_x[tr_va: va_te]
    # val_y = val_y + y[tr_va: va_te]
    test_x = test_x + new_x[tr_va:]
    test_y = test_y +y[tr_va:]

    # for test scaler y
    ori_y_train = ori_y[:tr_va]
    ori_y = ori_y[tr_va:]

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), ori_y_train, ori_y
    
def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def buy_and_hold_benchmark(stock_name, init_invest, n):
    df = pd.read_csv('./data/{}.csv'.format(stock_name))
    dates = df['DateTime'].astype("str")
    per_num_holding = init_invest // n
    num_holding = per_num_holding // df.iloc[0, 1:]
    balance_left = init_invest % n + ([per_num_holding for _ in range(n)] % df.iloc[0, 1:]).sum()
    buy_and_hold_portfolio_values = (df.iloc[:, 1:] * num_holding).sum(axis=1) + balance_left
    buy_and_hold_return = buy_and_hold_portfolio_values.iloc[-1] - init_invest
    return dates, buy_and_hold_portfolio_values, buy_and_hold_return


def plot_all(stock_name, model, daily_portfolio_value, env):
    """combined plots of plot_portfolio_transaction_history and plot_portfolio_performance_comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=100)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    dates, buy_and_hold_portfolio_values, buy_and_hold_return = buy_and_hold_benchmark(stock_name, env.init_invest, env.n_stock)
    agent_return = daily_portfolio_value[-1] - env.init_invest
    ax.set_title(f'{model} vs. Buy and Hold')
    dates = [datetime.strptime(d, '%Y%m%d').date() for d in dates]
    ###
    dates = dates[:-1]
    buy_and_hold_portfolio_values = buy_and_hold_portfolio_values[:-1]
    print(f'dates:     {len(dates)} \ndaily value:{len(daily_portfolio_value)}\nBH value:{len(buy_and_hold_portfolio_values)}')
    ###
    ax.plot(dates, daily_portfolio_value, color='green', label=f'{model} Total Return: ${agent_return:.2f}')
    ax.plot(dates, buy_and_hold_portfolio_values, color='blue', label=f'{stock_name} Buy and Hold Total Return: ${buy_and_hold_return:.2f}')
    ax.set_ylabel('Portfolio Value')

    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y%m%d'))
    # plt.xticks(pd.date_range('2018-1-02', '2019-08-22', freq='1m'))
    plt.xticks(pd.date_range(dates[0],dates[-1],freq='1m'))
    ax.legend()
    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def visualize_portfolio_val():
    """ visualize the portfolio_val file """
    with open('portfolio_val/201912141307-train.p', 'rb') as f:
        data = pickle.load(f)
    with open('portfolio_val/201912042043-train.p', 'rb') as f:
        data0 = pickle.load(f)
    print(sum(data) / 4000)
    print('data>>>', len(data))
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), dpi=100)

    ax[0].plot(data0, linewidth=1)
    ax[0].set_title('DQN Training Performance: 2000 episodes', fontsize=24)
    ax[0].set_xlabel('episode', fontsize=24)
    ax[0].set_ylabel('final portfolio value', fontsize=24)
    ax[0].tick_params(axis='both', labelsize=12)

    ax[1].plot(data, linewidth=1)
    ax[1].set_title('DQN Training Performance: 4000 episodes', fontsize=24)
    ax[1].set_xlabel('episode', fontsize=24)
    ax[1].set_ylabel('final portfolio value', fontsize=24)
    ax[1].tick_params(axis='both', labelsize=12)

    plt.show()


# visualize_portfolio_val()