from collections import deque
import random
import numpy as np
from model import dnn, conv1d, lstm, transformer


class DQNAgent(object):
    """ A simple Deep Q agent """

    def __init__(self, state_size, action_size, mode, model_type):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        # self.epsilon = 1.0  # exploration rate
        self.epsilon = 0.10  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        if model_type == 'dnn':
            self.model = dnn(state_size, action_size)
        if model_type == 'conv1d':
            self.model = conv1d(state_size, action_size)
        if model_type == 'lstm':
            self.model = lstm(state_size, action_size)
        if model_type == 'transformer':
            self.model = transformer(state_size, action_size)
        self.mode=mode

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.mode == 'train':
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=32):
        """ vectorized implementation; 30x speed up compared with for loop """
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([tup[0][0] for tup in minibatch])
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.array([tup[3][0] for tup in minibatch])
        done = np.array([tup[4] for tup in minibatch])

        # Q(s', a)
        target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.model.predict(states)
        # make the agent to approximately map the current state to future discounted reward
        target_f[range(batch_size), actions] = target

        self.model.fit(states, target_f, epochs=1, verbose=1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        # self.model.save_weights(name,include_optimizer=False)
        self.model.save_weights(name)
