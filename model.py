# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten, BatchNormalization, MaxPooling1D
from tensorflow.keras.optimizers import Adam


def dnn(n_obs, n_action):
    """ A multi-layer perceptron """
    # print('\n',n_obs[0],'\n')
    model = Sequential()
    model.add(Dense(units=1024, input_shape=[n_obs[1]], activation="relu"))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(n_action, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model


def conv1d(n_obs, n_action):
    kernel_size=2
    strides=1
    padding = 'same'
    model = Sequential()
    model.add(Conv1D(filters = 256, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu',input_shape=(n_obs[1],n_obs[2])))
    model.add(Conv1D(filters = 256, kernel_size=kernel_size, strides=strides, padding=padding, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(n_action, activation="softmax"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model

def lstm(n_obs, n_action):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,input_shape=(n_obs[1],n_obs[2])))
    # model.add(LSTM(128, dropout=0.2, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(n_action, activation="softmax"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model