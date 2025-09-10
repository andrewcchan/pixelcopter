from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import os
import argparse


game = Pixelcopter(width=48, height=48)
p = PLE(game, fps=30, display_screen=True)
p.init()


from sac_agent import SACAgent


# --- Save, Record, Evaluate ---
def record_video(agent, filename="pixelcopter_agent_sac.mp4", max_steps=1000):
    p.display_screen = True
    p.reset_game()
    state = p.getGameState()
    frames = []
    for t in range(max_steps):
        if p.game_over():
            break
        action = agent.select_action(state, evaluate=True)
        p.act(action)
        frame = p.getScreenRGB()
        frames.append(frame)
        state = p.getGameState()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved video to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Train or record a SAC agent for Pixelcopter.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "record"],
                        help="Mode to run the script in. 'train' will train a new agent, "
                             "'record' will load a pre-trained agent and record a video.")
    parser.add_argument("--checkpoint", type=str, default="sac_checkpoint.pth",
                        help="Path to the checkpoint file to load or save.")
    args = parser.parse_args()


    # --- Hyperparameters for SAC ---
    lr = 0.0003
    gamma = 0.99
    buffer_size = 100000
    batch_size = 256
    tau = 0.005
    alpha = 0.1  # Lowered alpha
    target_update_interval = 1
    learning_starts = 2000 # Increased exploration phase
    hidden_dim = 256 # Explicitly define network size

    # --- Initialization ---
    action_set = p.getActionSet()
    state_dim = len(p.getGameState().keys())
    action_dim = len(action_set)

    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, allowed_actions=action_set,
                     lr=lr, gamma=gamma, buffer_size=buffer_size, tau=tau, alpha=alpha)

    if args.mode == "record":
        if os.path.exists(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}")
            agent.load(args.checkpoint)
            record_video(agent, filename="pixelcopter_agent_sac_loaded.mp4")
        else:
            print(f"Checkpoint file not found at {args.checkpoint}. Please train the agent first.")
        return

    # --- Training Logic (only for mode='train') ---
    start_episode = 1
    # Let's start fresh
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)

    # --- Training Loop ---
    total_steps = 0
    episode_rewards = []
    for i_episode in range(start_episode, 401): # Reduced training time
        p.reset_game()
        state = p.getGameState()
        episode_reward = 0
        done = False

        while not done:
            total_steps += 1
            if total_steps < learning_starts:
                action = random.choice(action_set)
            else:
                action = agent.select_action(state)

            reward = p.act(action)
            # Clip rewards to be between -1 and 1, a common practice
            reward = np.clip(reward, -1, 1)

            next_state = p.getGameState()
            done = p.game_over()

            agent.store_transition(state, action, reward, next_state, done)

            if total_steps >= learning_starts and total_steps % 4 == 0: # Update every 4 steps
                agent.update(batch_size, target_update_interval, total_steps)

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        if i_episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {i_episode} | Avg Reward (last 10): {avg_reward:.2f} | Alpha: {agent.log_alpha.exp().item():.4f}")

        if i_episode % 50 == 0:
            print(f"--------------------------------------------------------")
            print(f"Saving checkpoint at episode {i_episode}")
            print(f"--------------------------------------------------------")
            agent.save(args.checkpoint, i_episode)

    print("Training complete.")
    record_video(agent)


if __name__ == '__main__':
    main()