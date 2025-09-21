import gymnasium as gym
import time

# It's important to install the package so gymnasium can find the environment
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "."])

import gym_projectile

def run_projectile_env():
    # The render_mode="human" will display the visualization
    env = gym.make('Projectile-v0', render_mode="human")

    # Reset the environment to get the initial observation
    observation, info = env.reset()

    # Set a fixed action for demonstration purposes
    # Action: [angle, initial_velocity]
    action = env.action_space.sample()
    print(f"Using action: angle={action[0]:.2f} degrees, velocity={action[1]:.2f} m/s")

    terminated = False
    while not terminated:
        # In this environment, the action is only used in the first step.
        # For subsequent steps, we can pass a dummy action.
        observation, reward, terminated, truncated, info = env.step(action)

        # The render method is called inside the step function when render_mode="human"

        if terminated:
            print(f"Episode finished. Final reward: {reward:.2f}")
            print(f"Distance to target: {info['distance']:.2f} meters")

    # Give some time for the plot to be seen before closing
    time.sleep(5)
    env.close()

if __name__ == "__main__":
    run_projectile_env()
