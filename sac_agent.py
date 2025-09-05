import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        logits = self.fc(state)
        return logits

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Outputs Q-value for each action
        )

    def forward(self, state):
        return self.fc(state)

class SACAgent:
    def __init__(self, state_dim, action_dim, allowed_actions, lr, gamma, buffer_size, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.allowed_actions = allowed_actions
        self.action_map = {i: a for i, a in enumerate(allowed_actions)}
        self.action_inv_map = {a: i for i, a in enumerate(allowed_actions)}

        # Actor
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critics
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.critic1_target = CriticNetwork(state_dim, action_dim)
        self.critic2_target = CriticNetwork(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

        # Learnable alpha
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -torch.tensor(action_dim, dtype=torch.float32).item()


        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # State Normalization
        self.running_state_mean = np.zeros(state_dim)
        self.running_state_std = np.ones(state_dim)
        self.state_count = 1e-4

        # Episode rewards tracking
        self.episode_rewards = []

    def normalize_state(self, state):
        state_vec = np.array(list(state.values()), dtype=np.float32)

        self.state_count += 1
        if self.state_count == 1:
            self.running_state_mean = state_vec
            # running_state_std is actually M2, the sum of squares of differences
            self.running_state_std = np.zeros(self.state_dim)
            return state_vec # Or a zero vector, but this is fine

        old_mean = self.running_state_mean.copy()
        self.running_state_mean += (state_vec - self.running_state_mean) / self.state_count
        self.running_state_std += (state_vec - old_mean) * (state_vec - self.running_state_mean)

        variance = self.running_state_std / (self.state_count - 1)
        std_dev = np.sqrt(variance + 1e-5)

        return (state_vec - self.running_state_mean) / std_dev

    def store_transition(self, state, action, reward, next_state, done):
        state_norm = self.normalize_state(state)
        next_state_norm = self.normalize_state(next_state)
        action_idx = self.action_inv_map[action]
        self.replay_buffer.store(state_norm, action_idx, reward, next_state_norm, done)
        if done:
            self.episode_rewards.append(np.sum(reward)) # Store total episode reward


    def select_action(self, state, evaluate=False):
        state_norm = self.normalize_state(state)
        state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if evaluate:
                action_idx = torch.argmax(probs, dim=-1)
            else:
                action_idx = dist.sample()

        return self.action_map[action_idx.item()]

    def update(self, batch_size, target_update_interval, total_steps):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        alpha = self.log_alpha.exp().detach()

        # Critic Update
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)

            q1_next = self.critic1_target(next_states)
            q2_next = self.critic2_target(next_states)
            min_q_next = torch.min(q1_next, q2_next)

            v_next = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * v_next

        q1 = self.critic1(states)
        q2 = self.critic2(states)

        critic_loss = F.mse_loss(q1.gather(1, actions), target_q) + F.mse_loss(q2.gather(1, actions), target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor and Alpha Update
        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            q1_val = self.critic1(states)
            q2_val = self.critic2(states)
            min_q = torch.min(q1_val, q2_val)

        actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).mean()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        if total_steps % target_update_interval == 0:
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_episode_rewards(self):
        return self.episode_rewards

    def save(self, filepath, episode):
        torch.save({
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'running_state_mean': self.running_state_mean,
            'running_state_std': self.running_state_std,
            'state_count': self.state_count
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.log_alpha.data = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.running_state_mean = checkpoint['running_state_mean']
        self.running_state_std = checkpoint['running_state_std']
        self.state_count = checkpoint['state_count']
        return checkpoint['episode']
