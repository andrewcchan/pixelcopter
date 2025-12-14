
import os
import torch
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from ple_wrapper import PLEPixelcopterGym

def train():
    # Create the environment
    # We use DummyVecEnv because VecNormalize requires a VecEnv
    env = DummyVecEnv([lambda: Monitor(PLEPixelcopterGym())])

    # Normalize observations (crucial for PPO) and rewards (optional, but often good)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Check if we can use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device=device
    )

    # Train the agent
    # Increased steps to ensure better convergence
    TOTAL_TIMESTEPS = 500000
    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Save the model
    model_path = "pixelcopter_ppo_model"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # Save the normalization stats
    env.save("vec_normalize.pkl")
    print("Saved normalization stats to vec_normalize.pkl")

def evaluate_and_record():
    print("Recording video of the trained agent...")

    # We need an environment with render_mode='rgb_array' for video
    # We must wrap it exactly as during training to use the stats
    eval_env = DummyVecEnv([lambda: PLEPixelcopterGym(render_mode='rgb_array')])

    # Load the saved normalization stats
    # We disable norm_reward because we want to see the real reward in evaluation
    eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
    eval_env.training = False # Do not update stats during evaluation
    eval_env.norm_reward = False

    # Load model
    model = PPO.load("pixelcopter_ppo_model", env=eval_env)

    frames = []
    obs = eval_env.reset()
    done = False
    total_reward = 0

    # Record until game over or a very high limit
    # "Win" condition usually means surviving a long time or reaching a high score.
    # We set a high limit (e.g. 5000 steps).
    max_steps = 5000
    for i in range(max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = eval_env.step(action)

        # In VecEnv, terminated is an array of booleans.
        # But wait, VecEnv step returns (obs, rewards, dones, infos)
        # 'terminated' here is actually 'dones' (terminated or truncated)

        total_reward += reward[0]

        # Capture frame
        # render() in VecEnv might tricky.
        # But our env supports render().
        # We can call render on the wrapped env.
        frame = eval_env.envs[0].render()
        if frame is not None:
            frames.append(frame)

        if terminated[0]:
            print(f"Episode finished at step {i+1}")
            break

    print(f"Evaluation episode reward: {total_reward}")
    video_path = "pixelcopter_agent.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Saved video to {video_path}")

if __name__ == "__main__":
    train()
    evaluate_and_record()
