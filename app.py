from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import imageio


game = Pixelcopter(width=48, height=48)
p = PLE(game, fps=30, display_screen=True)
p.init()


# Policy Gradient Agent using REINFORCE
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
        self.baseline_alpha = 0.99  # running average baseline
        self.entropy_beta = 0.01    # entropy regularization strength

    def normalize_state(self, state):
        # Update running mean and std, then normalize
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
        # Baseline: running mean of returns
        mean_return = returns.mean().item()
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * mean_return
        baseline_tensor = torch.full_like(returns, self.baseline)
        advantages = returns - baseline_tensor
        # Normalize advantages
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

# ...existing code...
action_set = p.getActionSet()

reward = 0.0
nb_episodes = 1000
max_steps = 1000
action_set = p.getActionSet()
state_keys = list(p.getGameState().keys())
agent = PolicyGradientAgent(allowed_actions=action_set, state_keys=state_keys)

episode_rewards = []
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
    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}: Total Reward = {total_reward}")


# Save the trained weights
torch.save(agent.policy.state_dict(), "pixelcopter_policy.pt")
print("Saved policy weights to pixelcopter_policy.pt")

# Save a video of the trained agent
def record_video(agent, filename="pixelcopter_agent.mp4", max_steps=1000):
    p.display_screen = True
    p.reset_game()
    state = p.getGameState()
    frames = []
    for t in range(max_steps):
        if p.game_over():
            break
        action = agent.select_action(state)
        p.act(action)
        frame = p.getScreenRGB()
        frames.append(frame)
        state = p.getGameState()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename}")

record_video(agent)

# Load and use the trained weights for evaluation
def evaluate_agent(weights_path="pixelcopter_policy.pt", episodes=5, max_steps=1000):
    eval_agent = PolicyGradientAgent(allowed_actions=action_set, state_keys=state_keys)
    eval_agent.policy.load_state_dict(torch.load(weights_path))
    eval_agent.policy.eval()
    rewards = []
    for ep in range(episodes):
        p.reset_game()
        state = p.getGameState()
        total_reward = 0
        for t in range(max_steps):
            if p.game_over():
                break
            action = eval_agent.select_action(state)
            reward = p.act(action)
            total_reward += reward
            state = p.getGameState()
        rewards.append(total_reward)
        print(f"[EVAL] Episode {ep+1}: Total Reward = {total_reward}")
    print(f"[EVAL] Mean reward over {episodes} episodes: {np.mean(rewards)}")

evaluate_agent()