
import os
import argparse
import numpy as np
import torch
import imageio
import gymnasium as gym
from gymnasium import spaces

# Third-party imports
from ple.games.pixelcopter import Pixelcopter
from ple import PLE

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm


# -----------------------------------------------------------------------------
# Environment Wrapper
# -----------------------------------------------------------------------------

class PixelcopterEnv(gym.Env):
    def __init__(self, config=None):
        self.width = 48
        self.height = 48
        # Using a dummy driver for headless environments, but allow overriding
        if os.environ.get("SDL_VIDEODRIVER") is None:
             # On a headless server (like this sandbox), we might want to default to dummy if not set.
             # But on a local MacBook, we might want the window.
             # The training script sets this via runtime_env, so we can leave it flexible here.
             pass

        self.game = Pixelcopter(width=self.width, height=self.height)
        # display_screen=False for speed during training, True might be needed for rendering
        display_screen = config.get("display_screen", False) if config else False
        self.p = PLE(self.game, fps=30, display_screen=display_screen)
        self.p.init()

        self.action_set = self.p.getActionSet()
        self.action_space = spaces.Discrete(len(self.action_set))

        # Determine observation space
        # We need to run one step to get the state keys and size
        self.p.reset_game()
        state = self.p.getGameState()
        self.state_keys = list(state.keys())
        # Sort keys to ensure consistent order
        self.state_keys.sort()

        low = -np.inf
        high = np.inf
        self.observation_space = spaces.Box(low=low, high=high, shape=(len(self.state_keys),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.p.reset_game()
        observation = self._get_obs()
        self.steps = 0
        info = {}
        return observation, info

    def step(self, action):
        # Map index to action
        act = self.action_set[action]
        reward = self.p.act(act)
        self.steps += 1

        observation = self._get_obs()
        terminated = self.p.game_over()
        truncated = self.steps >= 2000 # Force truncation if too long

        if terminated or truncated:
             # Just for debug/info
             # print(f"Episode finished. Steps: {self.steps}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
             pass

        info = {}
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        state = self.p.getGameState()
        obs = np.array([state[k] for k in self.state_keys], dtype=np.float32)
        return obs

    def render(self):
        # PLE returns rotated image for some reason, usually
        return self.p.getScreenRGB()


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------

def train_pixelcopter(num_iterations=10, checkpoint_path="pixelcopter_rllib_checkpoint"):
    # Initialize Ray
    # On a macbook, this works fine.
    # Set SDL_VIDEODRIVER to dummy to avoid pygame issues on headless systems (like this sandbox)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"SDL_VIDEODRIVER": "dummy"}})

    # Register the environment
    tune.register_env("pixelcopter_env", lambda config: PixelcopterEnv(config))

    # Configure PPO
    config = (
        PPOConfig()
        .environment("pixelcopter_env")
        .framework("torch")
        .env_runners(num_env_runners=1) # 1 worker is usually fine for local macbook
        .training(
            gamma=0.99,
            lr=0.0001,
            train_batch_size_per_learner=2000,
        )
        .resources(num_gpus=0) # Assume CPU for macbook unless mps is supported by ray (which is WIP)
    )

    # Build algorithm
    algo = config.build()

    print("Training started...")
    for i in range(num_iterations):
        result = algo.train()

        mean_reward = result.get('env_runners', {}).get('episode_reward_mean')
        if mean_reward is None:
             # Fallback for old API stack or if structure is different
             mean_reward = result.get('episode_reward_mean', 'N/A')

        print(f"Iteration {i}: mean_reward={mean_reward}")

    # Save the trained policy
    # Ensure the path is absolute or a valid URI
    final_checkpoint_path = algo.save_to_path(os.path.abspath(checkpoint_path))
    print(f"Final checkpoint saved at {final_checkpoint_path}")

    return algo, final_checkpoint_path


# -----------------------------------------------------------------------------
# Recording Function
# -----------------------------------------------------------------------------

def record_video(checkpoint_path="pixelcopter_rllib_checkpoint", video_filename="pixelcopter_rllib.mp4"):
    checkpoint_path = os.path.abspath(checkpoint_path)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"SDL_VIDEODRIVER": "dummy"}})

    # We need to register the env again just in case (if run separately)
    try:
        tune.register_env("pixelcopter_env", lambda config: PixelcopterEnv(config))
    except Exception:
        pass # Already registered

    print(f"Loading checkpoint from {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # Create the environment for recording
    env = PixelcopterEnv(config={"display_screen": True})

    obs, info = env.reset()
    frames = []

    done = False
    total_reward = 0
    steps = 0
    max_steps = 1000

    print("Recording video...")

    # Get the RLModule for inference
    module = algo.get_module("default_policy")

    while not done and steps < max_steps:
        # Compute action
        # Prepare batch
        obs_batch = torch.from_numpy(obs).unsqueeze(0).float()

        # Forward pass
        # The new API stack uses torch input dicts
        input_dict = {"obs": obs_batch}

        with torch.no_grad():
             # Forward inference
             action_dist_inputs = module.forward_inference(input_dict)
             # We need to sample from the distribution or take the argmax
             # PPO usually outputs logits for categorical
             logits = action_dist_inputs["action_dist_inputs"]
             dist = torch.distributions.Categorical(logits=logits)
             action = dist.sample().item()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Capture frame
        frame = env.render()
        frames.append(frame)

        done = terminated or truncated

    print(f"Episode finished. Total Reward: {total_reward}, Steps: {steps}")

    # Save video
    imageio.mimsave(video_filename, frames, fps=30)
    print(f"Saved video to {video_filename}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Record Pixelcopter Agent with RLlib")
    parser.add_argument("--mode", type=str, default="train_and_record", choices=["train", "record", "train_and_record"], help="Mode of operation")
    parser.add_argument("--checkpoint", type=str, default="pixelcopter_rllib_checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--video", type=str, default="pixelcopter_rllib.mp4", help="Output video filename")

    args = parser.parse_args()

    if args.mode in ["train", "train_and_record"]:
        _, checkpoint_path = train_pixelcopter(num_iterations=args.iterations, checkpoint_path=args.checkpoint)
        # Update checkpoint path for recording if we just trained
        args.checkpoint = checkpoint_path

    if args.mode in ["record", "train_and_record"]:
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint path {args.checkpoint} does not exist.")
        else:
            record_video(checkpoint_path=args.checkpoint, video_filename=args.video)
