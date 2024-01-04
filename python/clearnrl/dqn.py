# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import random

from env_grid_world import CustomEnv
from custom_replay_buffer import CustomBuffer
from network import TransformerQNetwork


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1

    # Algorithm specific arguments
    total_timesteps: int = 100000
    learning_rate: float = 2.5e-4
    buffer_size: int = 10000
    discount_factor: float = 0.99
    tau: float = 1.0  # the target network update rate
    target_network_frequency: int = 500
    batch_size: int = 128
    start_epsilon: float = 1
    end_epsilon: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10
    seq_len: int = 40


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    model_path = f"runs/{run_name}/{args.exp_name}.model"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")

    # env setup
    env = CustomEnv(4)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    print(env.observation_space, env.action_space)
    in_channels = env.observation_space.shape[0]
    out_channels = env.action_space.n

    q_network = TransformerQNetwork(
        in_channels, out_channels, args.seq_len).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = TransformerQNetwork(
        in_channels, out_channels, args.seq_len).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = CustomBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        args.seq_len,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    ideal_action_num = 0
    total_action_num = 0
    print_interval = args.total_timesteps // 20
    exploration_step = args.exploration_fraction * args.total_timesteps
    for global_step in range(1, args.total_timesteps + 1):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_epsilon, args.end_epsilon, exploration_step, global_step)
        if random.random() < epsilon:
            actions = env.action_space.sample()
        else:
            x = torch.Tensor(obs).to(device)
            data = rb.get_latest()
            observation = torch.roll(data.observations, -1, 1)
            observation[:, -1] = x
            action = torch.roll(data.actions, -1, 1)
            reward = torch.roll(data.rewards, -1, 1)
            total_time = torch.roll(data.total_times, -1, 1)
            q_values = q_network(observation, action, reward)
            q_values = q_values[:, -1]
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(
            actions)

        is_ideal_action = info["is_ideal_action"]
        ideal_action_num += int(is_ideal_action)
        total_action_num += 1
        ideal_action_rate = 100 * ideal_action_num / total_action_num
        if global_step % print_interval == 0:
            print(
                f"global_step: {global_step:07d}, ideal_action_rate: {ideal_action_rate:05.1f}%")
            writer.add_scalar("eval/ideal_rate",
                              ideal_action_rate, global_step)
            ideal_action_num = 0
            total_action_num = 0

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if truncation:
            real_next_obs = info["final_observation"]
        rb.add(obs, real_next_obs, actions, reward, termination, info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                curr_obs = data.observations
                curr_act = data.actions
                curr_rew = data.rewards

                next_obs = data.next_observations
                # 1つシフトすることでnextとする(最後の一個に入っているものはなんでも良いためこれで良い)
                next_act = torch.roll(curr_act, -1, 1)
                next_rew = torch.roll(curr_rew, -1, 1)
                with torch.no_grad():
                    target_q = target_network(next_obs, next_act, next_rew)
                    target_max, _ = target_q[:, -1].max(dim=1)
                    td_target = data.rewards[:, -1].flatten() + args.discount_factor * target_max * \
                        (1 - data.dones[:, -1].flatten())
                curr_q = q_network(curr_obs, curr_act, curr_rew)
                curr_q = curr_q[:, -1]
                old_val = curr_q.gather(1, data.actions[:, -1]).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values",
                                      old_val.mean().item(), global_step)
                    writer.add_scalar(
                        "charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data +
                        (1.0 - args.tau) * target_network_param.data
                    )

    torch.save(q_network.state_dict(), model_path)
    env.close()
    writer.close()
