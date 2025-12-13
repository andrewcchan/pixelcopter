
import os
import torch
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from ple_wrapper import PLEPixelcopterGym

# Create the environment
# We use a Monitor wrapper to track statistics
env = Monitor(PLEPixelcopterGym())

# Check if we can use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define the PPO model
# We use a MlpPolicy because the input is a vector (state)
# Hyperparameters can be tuned, but defaults are usually decent.
# We increase the number of steps to collect before update to ensure stability.
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
# The original script ran for 1000 episodes * 1000 steps max = 1M steps potential.
# We'll set a total timestep count. 500k should be enough to see good progress for this simple game.
TOTAL_TIMESTEPS = 200000
print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save the model
model_path = "pixelcopter_ppo_model"
model.save(model_path)
print(f"Saved model to {model_path}")

# Evaluation and Video Recording
print("Recording video of the trained agent...")
# We need an environment with render_mode='rgb_array' for video
eval_env = PLEPixelcopterGym(render_mode='rgb_array')

frames = []
obs, info = eval_env.reset()
done = False
total_reward = 0

# Record one episode or up to max steps
max_steps = 2000
for _ in range(max_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward

    # Capture frame
    frame = eval_env.render()
    if frame is not None:
        frames.append(frame)

    if terminated or truncated:
        break

print(f"Evaluation episode reward: {total_reward}")
video_path = "pixelcopter_agent.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"Saved video to {video_path}")
