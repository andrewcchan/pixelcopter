
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ple import PLE
from ple.games.pixelcopter import Pixelcopter

class PLEPixelcopterGym(gym.Env):
    """
    Gym wrapper for the PLE Pixelcopter environment.
    """
    def __init__(self, render_mode=None):
        super(PLEPixelcopterGym, self).__init__()

        self.game = Pixelcopter(width=48, height=48)
        # Initialize PLE. display_screen must be True for video recording if needed,
        # but for training usually False is faster. However, PLE requires display_screen=True
        # to generate screens unless we hack it.
        # render_mode 'rgb_array' implies we might want the screen.
        self.display_screen = (render_mode == 'rgb_array' or render_mode == 'human')

        # The game allows "up" (flap). If we don't flap, gravity pulls down.
        # We can model this as Discrete(2). 0 -> None (noop), 1 -> Flap (action_set[0])
        self.action_space = spaces.Discrete(2)

        # Observation space
        # State keys: ['player_y', 'player_vel', 'player_dist_to_ceil', 'player_dist_to_floor',
        #              'next_gate_dist_to_player', 'next_gate_block_top', 'next_gate_block_bottom']
        # These are 7 continuous values.
        # We should check bounds.
        # y is 0-48. vel is approx -20 to 20?
        # Let's assume -inf to inf for safety, or better bounds if known.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.render_mode = render_mode
        self.state_keys = ['player_y', 'player_vel', 'player_dist_to_ceil', 'player_dist_to_floor',
                           'next_gate_dist_to_player', 'next_gate_block_top', 'next_gate_block_bottom']

        # We pass state_preprocessor=None because we handle observation extraction ourselves in _get_obs.
        # This prevents PLE from trying to preprocess state during init.
        self.ple = PLE(self.game, fps=30, display_screen=self.display_screen)
        self.ple.init()

        # Action space: 0 (do nothing), 1 (flap)
        # Pixelcopter actions: 119 (up/flap) or None.
        # But PLE map: action_set = self.ple.getActionSet()
        # usually [119] for Pixelcopter (key 'w' is 119) and None is implicit if no action passed?
        # Actually PLE.act(action) takes the key code.
        self.action_set = self.ple.getActionSet()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ple.reset_game()
        # We need to step once or get state? PLE reset_game resets internal state.
        # getGameState returns the state.
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        # Map action index to PLE action
        # 0 -> None (NOOP)
        # 1 -> self.action_set[0] (FLAP)

        ple_action = None
        if action == 1:
            ple_action = self.action_set[0]

        # PLE returns reward for the step
        reward = self.ple.act(ple_action)

        # Check if game over
        terminated = self.ple.game_over()
        truncated = False # Pixelcopter usually doesn't time out unless we enforce it.

        observation = self._get_obs()
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        state = self.ple.getGameState()
        # Convert dict to array in fixed order
        obs = np.array([state[k] for k in self.state_keys], dtype=np.float32)
        return obs

    def render(self):
        if self.render_mode == 'rgb_array':
            # PLE getScreenRGB() returns numpy array
            return self.ple.getScreenRGB()
        return None

    def close(self):
        # self.ple.quit() # PLE doesn't have quit?
        pass
