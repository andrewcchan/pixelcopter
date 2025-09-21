import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class ProjectileEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, mass=1.0):
        super(ProjectileEnv, self).__init__()

        # The mass parameter is included for future extensions (e.g., air resistance),
        # but it does not affect projectile motion in a vacuum.
        self.mass = mass
        self.g = 9.81  # Acceleration due to gravity
        self.dt = 1 / self.metadata['render_fps']  # Time step

        # Action space: [angle (degrees), initial_velocity (m/s)]
        self.action_space = spaces.Box(
            low=np.array([0, 5]),
            high=np.array([90, 25]),
            dtype=np.float32
        )

        # Observation space: [x, y, vx, vy, target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, -np.inf, -np.inf, 0, 0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 0]),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.trajectory_line = None
        self.projectile_point = None
        self.target_point = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.state = np.zeros(4, dtype=np.float32)  # x, y, vx, vy

        # Randomize target position
        self.target_x = self.np_random.uniform(20, 100)
        self.target_y = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if self.t == 0:
            angle_deg, initial_velocity = action
            angle_rad = np.deg2rad(angle_deg)
            self.state[2] = initial_velocity * np.cos(angle_rad)  # vx
            self.state[3] = initial_velocity * np.sin(angle_rad)  # vy

        # Update projectile state
        self.state[0] += self.state[2] * self.dt  # x
        self.state[1] += self.state[3] * self.dt  # y
        self.state[3] -= self.g * self.dt         # vy

        self.t += 1

        # Check if episode is done
        terminated = self.state[1] < 0

        reward = 0
        if terminated:
            distance = np.sqrt((self.state[0] - self.target_x)**2)
            reward = -distance

        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([self.state, [self.target_x, self.target_y]]).astype(np.float32)

    def _get_info(self):
        return {"distance": np.sqrt((self.state[0] - self.target_x)**2)}

    def render(self):
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, 120)
            self.ax.set_ylim(0, 60)
            self.trajectory_line, = self.ax.plot([], [], 'b-')
            self.projectile_point, = self.ax.plot([], [], 'ro')
            self.target_point, = self.ax.plot([], [], 'gx', markersize=10)
            self.x_data, self.y_data = [], []

        self.x_data.append(self.state[0])
        self.y_data.append(self.state[1])

        self.trajectory_line.set_data(self.x_data, self.y_data)
        self.projectile_point.set_data([self.state[0]], [self.state[1]])
        self.target_point.set_data([self.target_x], [self.target_y])

        self.ax.set_xlim(0, max(120, self.target_x + 20))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self.dt)

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None
