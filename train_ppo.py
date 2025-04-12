import gym
import numpy as np
from gym import spaces
from perlin_noise import perlin_noise
from flood_fill import flood_fill

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=100, scale=10, view_size=5, max_steps=1000):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.scale = scale
        self.view_size = view_size
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(view_size, view_size), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        self.reset()

    def reset(self):
        raw_map = perlin_noise(self.grid_size, self.grid_size, scale=self.scale)
        self.map = (raw_map > 0).astype(np.uint8)  # 1 = free, 0 = wall
        self.explored = np.zeros_like(self.map)

        # Random free start
        free_cells = np.argwhere(self.map == 1)
        idx = np.random.choice(len(free_cells))
        self.agent_pos = tuple(free_cells[idx])
        self.explored[self.agent_pos] = 1

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
        import matplotlib.colors as mcolors

        vis = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        vis[self.map == 1] = [1, 1, 1]           # White for unexplored
        vis[self.explored == 1] = [0.3, 0.6, 1]  # Light blue for explored
        vis[self.map == 0] = [0, 0, 0]           # Black for walls

        y, x = self.agent_pos
        vis[y, x] = [1, 0, 0]  # Red for agent

        explored_count = np.sum(self.explored[self.map == 1])
        total_free = np.sum(self.map == 1)
        percent = 100 * explored_count / total_free if total_free > 0 else 0

        plt.imshow(vis)
        plt.title(f"Exploration Progress: {percent:.2f}% covered")
        plt.axis('off')
        plt.pause(0.01)
        plt.clf()
