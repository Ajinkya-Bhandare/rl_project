import gym
import numpy as np
from gym import spaces
from utils import generate_grid, spawn_point

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=100, view_size=5, max_steps=1000):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.view_size = view_size
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(view_size, view_size), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        self.reset()

    def reset(self):
        self.map = generate_grid(self.grid_size)  # ← Improved map generation
        self.explored = np.zeros_like(self.map)

        self.agent_pos = tuple(spawn_point(self.map))  # ← Safer start location
        self.explored[self.agent_pos] = 1  # Mark start as explored

        self.steps = 0
        self.total_reward = 0
        return self._get_obs()

    def step(self, action):
        y, x = self.agent_pos
        if action == 0:    # UP
            new_pos = (y - 1, x)
        elif action == 1:  # DOWN
            new_pos = (y + 1, x)
        elif action == 2:  # LEFT
            new_pos = (y, x - 1)
        elif action == 3:  # RIGHT
            new_pos = (y, x + 1)

        reward = 0
        done = False

        # Check bounds and collisions
        if (0 <= new_pos[0] < self.grid_size and
            0 <= new_pos[1] < self.grid_size and
            self.map[new_pos] == 1):
            self.agent_pos = new_pos
            if self.explored[new_pos] == 0:
                reward = 1
                self.explored[new_pos] = 1
        else:
            reward = -10  # Penalty for hitting wall

        self.steps += 1
        self.total_reward += reward
        done = self.steps >= self.max_steps or np.all(self.explored[self.map == 1])

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        y, x = self.agent_pos
        half = self.view_size // 2
        obs = np.zeros((self.view_size, self.view_size), dtype=np.uint8)

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    obs[dy + half, dx + half] = self.map[ny, nx]
        return obs

    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        vis = self.map.copy().astype(float)
        y, x = self.agent_pos
        vis[y, x] = 0.5
        plt.imshow(vis, cmap='gray')
        plt.title("Grid World Map (Agent=0.5)")
        plt.pause(0.01)
        plt.clf()
