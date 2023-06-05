import numpy as np
import gym
from gym import spaces
import random

class MagicCubeEnv(gym.Env):
    def __init__(self):
        super(MagicCubeEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # 分别对应上下左右前后6个方向
        self.observation_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3)))
        self.reset()

    def reset(self):
        self.driver_pos = (0, 0, 0)
        self.passenger1_pos = (1, 1, 1)
        self.passenger2_pos = (2, 2, 2)
        self.destination_pos = (0, 2, 2)
        return self.driver_pos

    def step(self, action):
        dx, dy, dz = 0, 0, 0

        if action == 0:  # 上
            dx = -1
        elif action == 1:  # 下
            dx = 1
        elif action == 2:  # 左
            dy = -1
        elif action == 3:  # 右
            dy = 1
        elif action == 4:  # 前
            dz = -1
        elif action == 5:  # 后
            dz = 1

        new_driver_pos = (self.driver_pos[0] + dx, self.driver_pos[1] + dy, self.driver_pos[2] + dz)
        new_driver_pos = tuple(np.clip(np.array(new_driver_pos), 0, 2))

        reward = -1
        done = False

        if new_driver_pos == self.passenger1_pos or new_driver_pos == self.passenger2_pos:
            reward = -0.5

        if new_driver_pos == self.destination_pos:
            if self.passenger1_pos != self.destination_pos or self.passenger2_pos != self.destination_pos:
                reward = 10
            done = True

        self.driver_pos = new_driver_pos

        return self.driver_pos, reward, done, {}

