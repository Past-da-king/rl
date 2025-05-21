import numpy as np
import random

class QAgent:
    def __init__(self, num_x, num_y, max_k, num_actions, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_rate=0.001):
        self.num_x = num_x
        self.num_y = num_y
        self.max_k = max_k
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = self.epsilon_start

        # Initialize Q-table: (x, y, k, action)
        self.q_table = np.zeros((num_x, num_y, max_k + 1, num_actions))

    def choose_action(self, state):
        x, y, k = state
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.num_actions) # Explore
        else:
            return np.argmax(self.q_table[x-1, y-1, k]) # Exploit (x,y are 1-indexed, numpy array 0-indexed)

    def update_q_table(self, state, action, reward, next_state):
        x, y, k = state
        next_x, next_y, next_k = next_state

        old_q_value = self.q_table[x-1, y-1, k, action]
        max_future_q = np.max(self.q_table[next_x-1, next_y-1, next_k])

        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)
        self.q_table[x-1, y-1, k, action] = new_q_value

    def decay_epsilon(self, epoch):
        self.epsilon = max(self.epsilon_end, self.epsilon_start - epoch * self.epsilon_decay_rate)

    def get_greedy_action(self, state):
        x, y, k = state
        return np.argmax(self.q_table[x-1, y-1, k])