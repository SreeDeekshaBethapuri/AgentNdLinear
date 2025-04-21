# === envs/gridworld.py ===
import numpy as np
import random

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)

    def reset(self):
        self.state = (0, 0)
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.size, self.size))
        obs[self.state] = 1
        return obs.flatten()

    def step(self, action):
        x, y = self.state
        if action == 0 and x > 0:
            x -= 1  # up
        elif action == 1 and x < self.size - 1:
            x += 1  # down
        elif action == 2 and y > 0:
            y -= 1  # left
        elif action == 3 and y < self.size - 1:
            y += 1  # right
        self.state = (x, y)
        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal
        return self._get_obs(), reward, done

    def sample_action(self):
        return random.randint(0, 3)