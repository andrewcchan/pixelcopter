from custom_env import CustomPixelcopter
from ple import PLE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import random

# Initialize custom environment
game = CustomPixelcopter(width=256, height=256)
p = PLE(game, fps=30, display_screen=False, force_fps=True)
p.init()

# Define Policy Network (same as app.py)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

class PolicyGradientAgent:
    def __init__(self, allowed_actions, state_keys, lr=1e-3):
        self.allowed_actions = allowed_actions
        self.state_keys = state_keys
        self.action_map = {i: a for i, a in enumerate(allowed_actions)}
        self.action_inv_map = {a: i for i, a in enumerate(allowed_actions)}
        self.policy = PolicyNetwork(len(state_keys), len(allowed_actions))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.running_state_mean = np.zeros(len(state_keys))
        self.running_state_std = np.ones(len(state_keys))
        self.state_count = 0
        self.baseline = 0.0
        self.baseline_alpha = 0.99
        self.entropy_beta = 0.01

    def normalize_state(self, state):
        state_vec = np.array([state[k] for k in self.state_keys], dtype=np.float32)
        self.state_count += 1
        self.running_state_mean = self.running_state_mean * (1 - 1/self.state_count) + state_vec * (1/self.state_count)
        self.running_state_std = self.running_state_std * (1 - 1/self.state_count) + ((state_vec - self.running_state_mean) ** 2) * (1/self.state_count)
        std = np.sqrt(self.running_state_std + 1e-8)
        return (state_vec - self.running_state_mean) / std

    def select_action(self, state):
        state_vec = self.normalize_state(state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)
        probs = self.policy(state_tensor)
        m = torch.distributions.Categorical(probs)
        action_idx = m.sample()
        self.log_probs.append(m.log_prob(action_idx))
        self.entropies.append(m.entropy())
        return self.action_map[action_idx.item()]

    def record_reward(self, reward):
        self.rewards.append(reward)

    def finish_episode(self, gamma=0.99):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) == 0: return # Avoid error on empty episode

        mean_return = returns.mean().item()
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * mean_return
        baseline_tensor = torch.full_like(returns, self.baseline)
        advantages = returns - baseline_tensor
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        loss = -torch.sum(log_probs * advantages) - self.entropy_beta * torch.sum(entropies)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.entropies = []
        self.rewards = []

# Training Loop
action_set = p.getActionSet()
state_keys = list(p.getGameState().keys())
print("State keys:", state_keys)
agent = PolicyGradientAgent(allowed_actions=action_set, state_keys=state_keys)

nb_episodes = 50
max_steps = 2000

for episode in range(nb_episodes):
    p.reset_game()
    state = p.getGameState()
    total_reward = 0
    for t in range(max_steps):
        if p.game_over():
            break
        action = agent.select_action(state)
        reward = p.act(action)
        agent.record_reward(reward)
        total_reward += reward
        state = p.getGameState()

    agent.finish_episode()
    if episode % 10 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward}, Level: {game.level}")

print("Training check complete. Generating video...")

# Record Video
p.display_screen = True
p.reset_game()
state = p.getGameState()
frames = []
for t in range(1000):
    if p.game_over():
        break
    action = agent.select_action(state)
    p.act(action)
    frame = p.getScreenRGB()
    frames.append(frame)
    state = p.getGameState()

imageio.mimsave("custom_agent_demo.mp4", frames, fps=30)
print("Saved video to custom_agent_demo.mp4")
