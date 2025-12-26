import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=10, maze_layout=None):
        super().__init__()
        self.size = size

        # 0: Empty, 1: Wall, 2: Agent, 3: Goal
        self.observation_space = spaces.Box(low=0, high=3, shape=(1, size, size), dtype=np.float32)
        self.action_space = spaces.Discrete(4) # Up, Right, Down, Left

        self.maze_layout = maze_layout
        self.fixed_layout = None

        self.agent_pos = None
        self.goal_pos = None
        self.grid = None

    def _generate_maze(self):
        # If a fixed layout is set, use it
        if self.fixed_layout is not None:
             return self.fixed_layout.copy(), (1, 1), (self.size-2, self.size-2)

        # Simple recursive backtracker for maze generation
        h, w = self.size, self.size
        grid = np.ones((h, w), dtype=np.int8) # Start with walls

        # Starting point
        start_row, start_col = 1, 1
        grid[start_row, start_col] = 0

        stack = [(start_row, start_col)]

        while stack:
            r, c = stack[-1]
            neighbors = []

            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < h-1 and 1 <= nc < w-1 and grid[nr, nc] == 1:
                    neighbors.append((nr, nc, dr//2, dc//2))

            if neighbors:
                nr, nc, wr, wc = random.choice(neighbors)
                grid[r+wr, c+wc] = 0
                grid[nr, nc] = 0
                stack.append((nr, nc))
            else:
                stack.pop()

        return grid, (1, 1), (h-2, w-2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "layout" in options:
             self.grid, self.agent_pos, self.goal_pos = options["layout"].copy(), (1, 1), (self.size-2, self.size-2)
        else:
             self.grid, self.agent_pos, self.goal_pos = self._generate_maze()

        self.grid[self.goal_pos] = 0
        self.grid[self.agent_pos] = 0

        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.grid.copy()
        obs[self.agent_pos] = 2
        obs[self.goal_pos] = 3
        return obs[np.newaxis, :, :].astype(np.float32)

    def step(self, action):
        # 0: Up, 1: Right, 2: Down, 3: Left
        row, col = self.agent_pos
        dr, dc = 0, 0
        if action == 0: dr = -1
        elif action == 1: dc = 1
        elif action == 2: dr = 1
        elif action == 3: dc = -1

        new_row, new_col = row + dr, col + dc

        reward = -0.05 # Smaller step penalty
        terminated = False
        truncated = False

        # Check bounds and walls
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            if self.grid[new_row, new_col] != 1:
                self.agent_pos = (new_row, new_col)

        # Check goal
        if self.agent_pos == self.goal_pos:
            reward = 10.0 # High reward for goal
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        print("\n" + "-" * self.size)
        grid = self.grid.copy()
        grid[self.agent_pos] = 2
        grid[self.goal_pos] = 3

        for r in range(self.size):
            line = ""
            for c in range(self.size):
                if grid[r, c] == 1: line += "#"
                elif grid[r, c] == 0: line += " "
                elif grid[r, c] == 2: line += "A"
                elif grid[r, c] == 3: line += "G"
            print(line)
        print("-" * self.size + "\n")


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Simplified model
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.fc_input_dim = 16 * input_dim * input_dim
        self.fc_common = nn.Linear(self.fc_input_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_common(x))

        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value

def train():
    maze_size = 5 # Even smaller maze
    env = MazeEnv(size=maze_size)

    # Train on RANDOM mazes
    print("Training on Random Mazes...")
    env.fixed_layout = None # Ensure random generation

    model = ActorCritic(maze_size, 4)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    num_episodes = 2000
    gamma = 0.95

    for episode in range(num_episodes):
        state, _ = env.reset() # Random layout
        done = False
        log_probs = []
        values = []
        rewards = []
        entropies = []

        steps = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs, value = model(state_tensor)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            entropies.append(dist.entropy())

            state = next_state
            steps += 1

            if steps >= 100:
                done = True

        # Calculate returns
        returns = []
        R = 0
        if not terminated:
             with torch.no_grad():
                 _, next_val = model(torch.FloatTensor(state).unsqueeze(0))
                 R = next_val.item()

        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()
        if values.dim() == 0: values = values.unsqueeze(0)

        advantage = returns - values.detach()

        actor_loss = -(torch.stack(log_probs) * advantage).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -torch.stack(entropies).mean()

        loss = actor_loss + 0.5 * critic_loss + 0.05 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}, Steps: {len(rewards)}")

    return model

def evaluate(model):
    print("\nStarting Evaluation on New Mazes...")
    maze_size = 5
    env = MazeEnv(size=maze_size)

    # Generate 5 NEW random mazes
    eval_mazes = []
    for _ in range(5):
        layout, _, _ = env._generate_maze()
        eval_mazes.append(layout)

    success_count = 0

    for i, layout in enumerate(eval_mazes):
        print(f"\nEval Maze {i+1}:")
        obs, _ = env.reset(options={"layout": layout})
        env.render()

        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 50:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs, _ = model(state_tensor)
                action = torch.argmax(probs).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        if terminated:
            print("Status: SUCCESS")
            success_count += 1
        else:
            print("Status: FAILED")
        print(f"Reward: {total_reward:.2f}")

    print(f"\nEvaluation Complete. Success Rate: {success_count}/5")

if __name__ == "__main__":
    trained_model = train()
    evaluate(trained_model)
