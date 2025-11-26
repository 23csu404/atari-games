import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_shape, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.Q = np.zeros(state_shape + (n_actions,))
        self.actions = list(range(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state])
        self.Q[state + (action,)] = (1 - self.alpha) * self.Q[state + (action,)] + \
                                    self.alpha * (reward + self.gamma * best_next)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
