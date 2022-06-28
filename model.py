# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
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

def conv1d():
    model = Sequential()
    model.add(lstm(units=1024, input_dim=n_obs, activation="relu"))
    model.add(Dropout(0.3))
    model.add(lstm(units=1024, activation="relu"))
    model.add(Dense(n_action, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model

def lstm():
    model = Sequential()
    model.add(lstm(units=1024, input_dim=n_obs, activation="relu"))
    model.add(lstm(units=1024, activation="relu"))
    model.add(Dense(n_action, activation="linear"))
    model.add(Dropout(0.3))
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    print(model.summary())
    return model