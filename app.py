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

# --- Main Training Loop ---
action_set = p.getActionSet()
agent = SimpleBanditAgent(actions=action_set, epsilon=0.1)

nb_episodes = 1000
max_steps_per_episode = 1000
episode_rewards = []

for episode in range(nb_episodes):
    p.reset_game()
    total_reward = 0
    for step in range(max_steps_per_episode):
        if p.game_over():
            break

        # Select an action and perform it
        action = agent.select_action()
        reward = p.act(action)
        total_reward += reward

        # Update the agent
        agent.update(action, reward)

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{nb_episodes} | Total Reward: {total_reward}")

print("\\nTraining finished.")
print("Final action values:", agent.action_values)


# --- Evaluation and Video Recording ---

def record_video(agent, filename="pixelcopter_agent_bandit.mp4", max_steps=1000):
    """Records a video of the agent playing one episode."""
    p.display_screen = True
    agent.epsilon = 0  # Turn off exploration for recording
    p.reset_game()
    frames = []
    for t in range(max_steps):
        if p.game_over():
            break
        action = agent.select_action()
        p.act(action)
        frame = p.getScreenRGB()
        frames.append(frame)
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename}")
    agent.epsilon = 0.1 # Reset epsilon

def evaluate_agent(agent, episodes=10, max_steps=1000):
    """Evaluates the agent's performance over several episodes."""
    agent.epsilon = 0.05  # Lower epsilon for evaluation
    rewards = []
    for ep in range(episodes):
        p.reset_game()
        total_reward = 0
        for t in range(max_steps):
            if p.game_over():
                break
            action = agent.select_action()
            reward = p.act(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"[EVAL] Episode {ep + 1}: Total Reward = {total_reward}")
    mean_reward = np.mean(rewards)
    print(f"\\n[EVAL] Mean reward over {episodes} episodes: {mean_reward}")
    agent.epsilon = 0.1 # Reset epsilon
    return mean_reward

# --- Run Evaluation and Record Video ---
print("\\n--- Evaluating Agent ---")
evaluate_agent(agent)

print("\\n--- Recording Video ---")
record_video(agent)