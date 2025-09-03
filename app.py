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


# Actor-Critic Networks for PPO
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, allowed_actions, lr, gamma, K_epochs, eps_clip, gae_lambda=0.95):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.action_map = {i: a for i, a in enumerate(allowed_actions)}
        self.action_inv_map = {a: i for i, a in enumerate(allowed_actions)}

        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        self.memory = []
        self.state_dim = state_dim
        self.running_state_mean = np.zeros(state_dim)
        self.running_state_std = np.ones(state_dim)
        self.state_count = 0

    def normalize_state(self, state):
        # Simple running mean and std normalization
        state_vec = np.array(list(state.values()), dtype=np.float32)
        self.state_count += 1
        self.running_state_mean = self.running_state_mean + (state_vec - self.running_state_mean) / self.state_count
        self.running_state_std = self.running_state_std + ((state_vec - self.running_state_mean)**2 - self.running_state_std) / self.state_count
        std = np.sqrt(self.running_state_std + 1e-5)
        return (state_vec - self.running_state_mean) / std

    def select_action(self, state):
        state_norm = self.normalize_state(state)
        state_tensor = torch.tensor(state_norm, dtype=torch.float32)

        with torch.no_grad():
            action_logits = self.actor(state_tensor)
            value = self.critic(state_tensor)

        dist = torch.distributions.Categorical(logits=action_logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        return self.action_map[action_idx.item()], state_tensor, action_idx, log_prob, value

    def store_transition(self, state, action_idx, log_prob, reward, done, value):
        self.memory.append((state, action_idx, log_prob, reward, done, value))

    def update(self):
        old_states, old_actions, old_logprobs, rewards, dones, old_values = zip(*self.memory)

        # Convert to tensor
        old_states = torch.squeeze(torch.stack(list(old_states), dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(list(old_actions), dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(list(old_logprobs), dim=0)).detach()
        old_values = torch.squeeze(torch.stack(list(old_values), dim=0)).detach()

        # Calculate advantages using GAE
        advantages = []
        last_advantage = 0
        last_value = old_values[-1]
        for i in reversed(range(len(rewards))):
            if dones[i]:
                mask = 0
            else:
                mask = 1
            delta = rewards[i] + self.gamma * last_value * mask - old_values[i]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            advantages.insert(0, last_advantage)
            last_value = old_values[i]

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = 0.5 * (state_values - returns).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []

    def evaluate(self, state, action):
        action_logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

# --- Hyperparameters ---
lr = 0.0003
gamma = 0.99
K_epochs = 10
eps_clip = 0.2
update_timestep = 4096

# --- Training ---
action_set = p.getActionSet()
state_keys = list(p.getGameState().keys())
state_dim = len(state_keys)
action_dim = len(action_set)

agent = PPOAgent(state_dim, action_dim, action_set, lr, gamma, K_epochs, eps_clip)

time_step = 0
episode_rewards = []

for i_episode in range(1, 2001):
    p.reset_game()
    state = p.getGameState()
    episode_reward = 0
    for t in range(1000):
        time_step += 1
        action, state_tensor, action_idx, log_prob, value = agent.select_action(state)
        reward = p.act(action)
        done = p.game_over()

        agent.store_transition(state_tensor, action_idx, log_prob, reward, done, value)

        episode_reward += reward

        if time_step % update_timestep == 0:
            agent.update()

        state = p.getGameState()
        if done:
            break

    episode_rewards.append(episode_reward)

    if i_episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {i_episode}\tAverage Reward: {avg_reward:.2f}")

final_avg_reward = np.mean(episode_rewards)
print(f"Training complete. Final average reward: {final_avg_reward:.2f}")

# --- Save, Record, Evaluate ---
def record_video(agent, filename="pixelcopter_agent_ppo.mp4", max_steps=1000):
    p.display_screen = True
    p.reset_game()
    state = p.getGameState()
    frames = []
    for t in range(max_steps):
        if p.game_over():
            break
        action, _, _, _, _ = agent.select_action(state)
        p.act(action)
        frame = p.getScreenRGB()
        frames.append(frame)
        state = p.getGameState()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename}")

record_video(agent)