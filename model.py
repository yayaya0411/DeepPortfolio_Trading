# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D,Flatten
from tensorflow.keras.optimizers import Adam


def dnn(n_obs, n_action):
    """ A multi-layer perceptron """
    model = Sequential()
    model.add(Dense(units=1024, input_dim=n_obs, activation="relu"))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(n_action, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model

def conv1d(n_obs, n_action):
    model = Sequential()
    model.add(lstm(units=1024, input_dim=n_obs, activation="relu"))
    model.add(Dropout(0.3))
    model.add(lstm(units=1024, activation="relu"))
    model.add(Dense(n_action, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model

def lstm(n_obs, n_action):
    model = Sequential()
    model.add(LSTM(64,return_sequences=True,input_shape=(20,10)))
    model.add(LSTM(128, dropout=0.2, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(n_action, activation="softmax"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model