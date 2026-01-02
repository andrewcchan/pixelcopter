
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from pixelcopter_env import PixelcopterEnv
import gymnasium as gym
import os

def train_pixelcopter():
    # Initialize Ray
    # On a macbook, this works fine.
    # Set SDL_VIDEODRIVER to dummy to avoid pygame issues on headless systems (like this sandbox)
    # This might not be strictly necessary on a macbook with a screen, but it ensures compatibility.
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
    for i in range(10): # Train for 10 iterations for demo
        result = algo.train()
        # pprint(result) # For debugging

        mean_reward = result.get('env_runners', {}).get('episode_reward_mean')
        if mean_reward is None:
             # Fallback for old API stack or if structure is different
             mean_reward = result.get('episode_reward_mean', 'N/A')

        print(f"Iteration {i}: mean_reward={mean_reward}")

        # Checkpoint if needed
        # if i % 10 == 0:
        #     checkpoint_dir = algo.save()
        #     print(f"Checkpoint saved at {checkpoint_dir}")

    # Save the trained policy
    # Ensure the path is absolute or a valid URI
    checkpoint_dir = algo.save_to_path(os.path.abspath("pixelcopter_rllib_checkpoint"))
    print(f"Final checkpoint saved at {checkpoint_dir}")

    return algo

if __name__ == "__main__":
    train_pixelcopter()
