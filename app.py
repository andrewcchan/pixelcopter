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


# Actor-Critic Networks
class ActorNetwork(nn.Module):
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

class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def __init__(self, allowed_actions, state_keys, lr=1e-4, gamma=0.99, clip_epsilon=0.2, ppo_epochs=4, batch_size=32):
        self.allowed_actions = allowed_actions
        self.state_keys = state_keys
        self.action_map = {i: a for i, a in enumerate(allowed_actions)}

        self.actor = ActorNetwork(len(state_keys), len(allowed_actions))
        self.critic = CriticNetwork(len(state_keys))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs

        self.memory = []

        self.running_state_mean = np.zeros(len(state_keys))
        self.running_state_std = np.ones(len(state_keys))
        self.state_count = 0

    def normalize_state(self, state):
        state_vec = np.array([state[k] for k in self.state_keys], dtype=np.float32)
        self.state_count += 1
        delta = state_vec - self.running_state_mean
        self.running_state_mean += delta / self.state_count
        delta2 = state_vec - self.running_state_mean
        self.running_state_std += delta * delta2
        std = np.sqrt(self.running_state_std / self.state_count + 1e-8)
        return (state_vec - self.running_state_mean) / std

    def select_action(self, state):
        state_vec = self.normalize_state(state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_tensor)
        m = torch.distributions.Categorical(probs)
        action_idx = m.sample()
        return self.action_map[action_idx.item()], action_idx, m.log_prob(action_idx)

    def store_transition(self, state, action_idx, log_prob, reward, done):
        self.memory.append((self.normalize_state(state), action_idx, log_prob, reward, done))

    def update(self):
        states, actions, old_log_probs, rewards, dones = zip(*self.memory)

        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        old_states = torch.tensor(np.array(states), dtype=torch.float32)
        old_actions = torch.tensor([a.item() for a in actions], dtype=torch.int64)
        old_log_probs = torch.tensor([lp.item() for lp in old_log_probs], dtype=torch.float32)

        for _ in range(self.ppo_epochs):
            log_probs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = nn.functional.mse_loss(state_values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.mean().backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.memory = []

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

# ...existing code...
action_set = p.getActionSet()

reward = 0.0
nb_episodes = 500
max_steps = 2000
update_timestep = 1000
action_set = p.getActionSet()
state_keys = list(p.getGameState().keys())
agent = PPOAgent(allowed_actions=action_set, state_keys=state_keys)

episode_rewards = []
time_step = 0

for episode in range(nb_episodes):
    p.reset_game()
    state = p.getGameState()
    total_reward = 0
    for t in range(max_steps):
        time_step += 1
        action, action_idx, log_prob = agent.select_action(state)
        reward = p.act(action)
        done = p.game_over()
        agent.store_transition(state, action_idx, log_prob, reward, done)

        if time_step % update_timestep == 0:
            agent.update()

        total_reward += reward
        state = p.getGameState()
        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}: Total Reward = {total_reward}")


# Save the trained weights
torch.save(agent.actor.state_dict(), "pixelcopter_actor.pt")
print("Saved actor weights to pixelcopter_actor.pt")

# Save a video of the trained agent
def record_video(agent, filename="pixelcopter_agent.mp4", max_steps=1000):
    p.display_screen = True
    p.reset_game()
    state = p.getGameState()
    frames = []
    for t in range(max_steps):
        if p.game_over():
            break
        action, _, _ = agent.select_action(state)
        p.act(action)
        frame = p.getScreenRGB()
        frames.append(frame)
        state = p.getGameState()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename}")

record_video(agent)

# Load and use the trained weights for evaluation
def evaluate_agent(weights_path="pixelcopter_actor.pt", episodes=5, max_steps=1000):
    eval_agent = PPOAgent(allowed_actions=action_set, state_keys=state_keys)
    eval_agent.actor.load_state_dict(torch.load(weights_path))
    eval_agent.actor.eval()
    rewards = []
    for ep in range(episodes):
        p.reset_game()
        state = p.getGameState()
        total_reward = 0
        for t in range(max_steps):
            if p.game_over():
                break
            action, _, _ = eval_agent.select_action(state)
            reward = p.act(action)
            total_reward += reward
            state = p.getGameState()
        rewards.append(total_reward)
        print(f"[EVAL] Episode {ep+1}: Total Reward = {total_reward}")
    print(f"[EVAL] Mean reward over {episodes} episodes: {np.mean(rewards)}")

evaluate_agent()