from typing import Callable
import torch
import gymnasium as gym
import numpy as np
import random


def evaluate(
    model_path: str,
    make_env: Callable,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    env = make_env()
    model = Model(2, 5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = env.reset()
    n_steps = 1000
    ideal_num = 0
    for _ in range(n_steps):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(torch.Tensor(obs).to(device).unsqueeze(0))
            action = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, reward, terminated, truncated, infos = env.step(action)
        is_ideal_action = int(infos.get("is_ideal_action"))
        ideal_num += is_ideal_action
        obs = next_obs

    ideal_rate = 100 * ideal_num / n_steps

    return ideal_rate
