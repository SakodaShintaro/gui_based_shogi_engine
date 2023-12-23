import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

kUp = 0
kRight = 1
kDown = 2
kLeft = 3
kClick = 4
kActionSize = 5


class CustomEnv(gym.Env):
    metadata = {"render_modes": ["console", "human"], "render_fps": 30}

    def __init__(self, grid_size: int):
        super().__init__()
        self.render_mode = "human"
        minimal_resolution = 36
        self.action_space = spaces.Discrete(kActionSize)

        # if grid_size < minimal_resolution, then scale up is needed
        self.scale = (minimal_resolution + grid_size - 1) // grid_size
        resolution = grid_size * self.scale
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(2, resolution, resolution), dtype=np.uint8)
        self.grid_size = grid_size
        self.position_self_x = random.randrange(self.grid_size)
        self.position_self_y = random.randrange(self.grid_size)
        self.position_goal_x = random.randrange(self.grid_size)
        self.position_goal_y = random.randrange(self.grid_size)

    def _make_observation(self):
        resolution = self.grid_size * self.scale
        observation = np.zeros((2, resolution, resolution), dtype=np.uint8)
        observation[0, self.position_self_x * self.scale:(self.position_self_x + 1) * self.scale,
                    self.position_self_y * self.scale:(self.position_self_y + 1) * self.scale] = 255
        observation[1, self.position_goal_x * self.scale:(self.position_goal_x + 1) * self.scale,
                    self.position_goal_y * self.scale:(self.position_goal_y + 1) * self.scale] = 255
        return observation

    def step(self, action):
        reward = -0.01
        terminated = False
        info = {"is_ideal_action": self._is_ideal_action(action)}
        if action == kUp:
            self.position_self_y -= 1
        elif action == kRight:
            self.position_self_x += 1
        elif action == kDown:
            self.position_self_y += 1
        elif action == kLeft:
            self.position_self_x -= 1
        elif action == kClick:
            if self.position_self_x == self.position_goal_x and self.position_self_y == self.position_goal_y:
                reward = 1
                self.position_goal_x = random.randrange(self.grid_size)
                self.position_goal_y = random.randrange(self.grid_size)
                terminated = True
        self.position_self_x = min(
            max(self.position_self_x, 0), self.grid_size - 1)
        self.position_self_y = min(
            max(self.position_self_y, 0), self.grid_size - 1)
        truncated = False
        return self._make_observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.position_goal_x = random.randrange(self.grid_size)
        self.position_goal_y = random.randrange(self.grid_size)
        return self._make_observation(), {}

    def render(self, mode="human"):
        self.render_mode = mode
        if mode == "console" or mode == 'human':
            print("state")
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
            grid[self.position_self_x, self.position_self_y] += 1
            grid[self.position_goal_x, self.position_goal_y] += 2
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if grid[i, j] == 0:
                        print(".", end="")
                    elif grid[i, j] == 1:
                        print("S", end="")
                    elif grid[i, j] == 2:
                        print("G", end="")
                    elif grid[i, j] == 3:
                        print("O", end="")
                print()

    def close(self):
        pass

    def _is_ideal_action(self, action):
        if self.position_self_x == self.position_goal_x and self.position_self_y == self.position_goal_y:
            return action == kClick
        else:
            dx = self.position_goal_x - self.position_self_x
            dy = self.position_goal_y - self.position_self_y
            if action == kUp:
                return dy < 0
            elif action == kRight:
                return dx > 0
            elif action == kDown:
                return dy > 0
            elif action == kLeft:
                return dx < 0
        return False


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = CustomEnv(4)
    check_env(env, warn=True)
