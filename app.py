from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import numpy as np
import random
import time
import imageio


game = Pixelcopter(width=48, height=48)
p = PLE(game, fps=30, display_screen=True)
p.init()


# Simple Multi-Armed Bandit Agent
class SimpleBanditAgent:
    def __init__(self, actions, epsilon=0.1):
        self.actions = actions
        self.epsilon = epsilon
        self.action_values = {a: 0 for a in actions}
        self.action_counts = {a: 0 for a in actions}

    def select_action(self):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            # Exploit
            return max(self.action_values, key=self.action_values.get)

    def update(self, action, reward):
        # Update the value of an action
        self.action_counts[action] += 1
        # Incremental update using sample averages
        self.action_values[action] = self.action_values[action] + \
            (1 / self.action_counts[action]) * (reward - self.action_values[action])

# This is where the new agent and training loop will go.