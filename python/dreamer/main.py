from utils import load_config
from dreamer import Dreamer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import os
from env_grid_world import CustomEnv
import gymnasium as gym


def get_env_infos(env):
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action_bool = True
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discrete_action_bool = False
        action_size = env.action_space.shape[0]
    else:
        raise Exception
    return obs_shape, discrete_action_bool, action_size

def main():
    config_file = "./config.yaml"
    config = load_config(config_file)

    env = CustomEnv(4)
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    log_dir = (
        "./runs/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
    )
    writer = SummaryWriter(log_dir)
    device = config.operation.device

    agent = Dreamer(
        obs_shape, discrete_action_bool, action_size, writer, device, config
    )
    agent.train(env)


if __name__ == "__main__":
    main()
