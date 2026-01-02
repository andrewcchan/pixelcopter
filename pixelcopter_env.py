
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import os

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
             print(f"Episode finished. Steps: {self.steps}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        info = {}
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        state = self.p.getGameState()
        obs = np.array([state[k] for k in self.state_keys], dtype=np.float32)
        return obs

    def render(self):
        # PLE returns rotated image for some reason, usually
        return self.p.getScreenRGB()
