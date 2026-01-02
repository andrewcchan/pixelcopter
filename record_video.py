
import gymnasium as gym
from pixelcopter_env import PixelcopterEnv
import torch
import numpy as np
import imageio
from ray.rllib.algorithms.algorithm import Algorithm
import ray
import os

def record_video(checkpoint_path="pixelcopter_rllib_checkpoint", video_filename="pixelcopter_rllib.mp4"):
    checkpoint_path = os.path.abspath(checkpoint_path)
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"SDL_VIDEODRIVER": "dummy"}})

    # Load the trained agent
    # We need to register the env again just in case
    from ray import tune
    tune.register_env("pixelcopter_env", lambda config: PixelcopterEnv(config))

    # We need to find the checkpoint file inside the directory
    # The `save_to_path` created a directory. RLlib usually saves a `checkpoint_000000` folder inside?
    # Or maybe it saved the algorithm state directly if using new API?
    # Let's assume the path provided is the one to load from.
    # However, `Algorithm.from_checkpoint` expects the folder containing the checkpoint or the checkpoint file.

    print(f"Loading checkpoint from {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # Create the environment for recording
    # We set display_screen=True to enable rendering, but we might need to handle the window issue
    # Since we are in headless, we rely on SDL_VIDEODRIVER=dummy and getScreenRGB
    env = PixelcopterEnv(config={"display_screen": True})

    obs, info = env.reset()
    frames = []

    done = False
    total_reward = 0
    steps = 0
    max_steps = 1000

    print("Recording video...")

    # Get the RLModule for inference
    # In the new API stack, we should use the RLModule directly or the EnvRunner
    # But compute_single_action is deprecated.
    # We can try to use the algo.compute_actions method or access the module.
    # However, since Algorithm.from_checkpoint restores the state, we should be able to just use the module.

    # Let's try to get the module
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

if __name__ == "__main__":
    record_video()
