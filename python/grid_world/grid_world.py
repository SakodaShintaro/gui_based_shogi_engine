import numpy as np
import random
from action import *


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class GridWorld:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.self_position = Position(random.randint(0, grid_size - 1),
                                      random.randint(0, grid_size - 1))
        self.goal_position = Position(random.randint(0, grid_size - 1),
                                      random.randint(0, grid_size - 1))

    def print(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.self_position.y][self.self_position.x] += 1
        grid[self.goal_position.y][self.goal_position.x] += 2
        for row in grid:
            print(''.join(map(str, row)))

    def step(self, action):
        if action == kClick:
            if self.self_position == self.goal_position:
                self.goal_position = Position(random.randint(0, self.grid_size - 1),
                                              random.randint(0, self.grid_size - 1))
                return True
            else:
                return False

        if action == kUp:
            self.self_position.y -= 1
        elif action == kRight:
            self.self_position.x += 1
        elif action == kDown:
            self.self_position.y += 1
        elif action == kLeft:
            self.self_position.x -= 1

        self.self_position.x = max(
            0, min(self.self_position.x, self.grid_size - 1))
        self.self_position.y = max(
            0, min(self.self_position.y, self.grid_size - 1))
        return False

    def state(self):
        state = np.zeros((2, self.grid_size, self.grid_size))
        state[0][self.self_position.y][self.self_position.x] = 1
        state[1][self.goal_position.y][self.goal_position.x] = 1
        return state

    def is_ideal_action(self, action):
        if self.self_position == self.goal_position:
            return action == kClick
        else:
            dx = self.goal_position.x - self.self_position.x
            dy = self.goal_position.y - self.self_position.y
            if action == kUp:
                return dy < 0
            elif action == kRight:
                return dx > 0
            elif action == kDown:
                return dy > 0
            elif action == kLeft:
                return dx < 0
        return False
