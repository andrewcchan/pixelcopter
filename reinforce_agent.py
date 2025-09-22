import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
        state_vec = np.array([state[k] for k in self.state_keys], dtype=np.float32)
        self.state_count += 1
        # More stable running mean/std update
        delta = state_vec - self.running_state_mean
        self.running_state_mean += delta / self.state_count
        delta2 = state_vec - self.running_state_mean
        self.running_state_std += delta * delta2

        std_dev = np.sqrt(self.running_state_std / (self.state_count - 1)) if self.state_count > 1 else np.ones(len(self.state_keys))
        return (state_vec - self.running_state_mean) / (std_dev + 1e-8)

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
        if not self.rewards:
            return

        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * returns.mean().item()
        advantages = returns - self.baseline

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

def train_reinforce(p, agent, nb_episodes, max_steps, callback=None):
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
        if callback:
            callback(episode, total_reward)
    return episode_rewards
