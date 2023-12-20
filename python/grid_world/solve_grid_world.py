import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from grid_world import GridWorld
from action import *


class NeuralNetwork(nn.Module):
    def __init__(self, h, w):
        super(NeuralNetwork, self).__init__()
        self.h = h
        self.w = w
        self.square = h * w

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_out_channels = [(2, 32), (32, 64), (64, 64), (64, 64)]
        for in_channels, out_channels in in_out_channels:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, 3, padding=1))

        hidden_dim = in_out_channels[-1][1] * h * w
        self.linear_policy = nn.Linear(hidden_dim, kActionSize)
        self.linear_value = nn.Linear(hidden_dim, 1)

        s2 = self.square * self.square
        self.policy_embedding = nn.Embedding(s2, kActionSize)
        self.value_embedding = nn.Embedding(s2, 1)
        nn.init.constant_(self.policy_embedding.weight, 0)
        nn.init.constant_(self.value_embedding.weight, 0)

    def forward(self, x):
        # x : [bs, 2, h, w]
        # NN推論する場合
        # for layer in self.conv_layers:
        #     x = layer(x)
        #     x = nn.ReLU()(x)
        # x = x.flatten()
        # policy = self.linear_policy(x)
        # value = self.linear_value(x)

        # テーブルベースで考える場合
        # 1になっているindexを取得する
        x = x.flatten(2)  # [bs, 2, h * w]
        dim0 = x[:, 0].nonzero(as_tuple=False)
        dim1 = x[:, 1].nonzero(as_tuple=False)
        dim0 = dim0[:, 1]
        dim1 = dim1[:, 1]
        index = dim0 * self.square + dim1
        policy = self.policy_embedding(index)
        value = self.value_embedding(index)

        return policy, value


def main():
    kGridSize = 4
    grid = GridWorld(kGridSize)

    network = NeuralNetwork(kGridSize, kGridSize)

    optimizer = optim.SGD(network.parameters(), 1.0)

    kGamma = 0.9  # 割引率

    is_ideal_actions = deque()
    kWindowSize = 200
    ideal_action_num = 0

    f = open("grid_world_log.tsv", "w")

    for i in range(20000):
        print("")
        grid.print()

        state = grid.state()
        state = torch.tensor(state, dtype=torch.float32)

        policy_logit, value = network(state.unsqueeze(0))
        policy_logit = policy_logit[0]
        value = value[0]
        policy = torch.softmax(policy_logit, 0)
        action = torch.multinomial(policy, 1).item()

        is_ideal_action = grid.is_ideal_action(action)
        is_ideal_actions.append(is_ideal_action)
        ideal_action_num += is_ideal_action
        if len(is_ideal_actions) > kWindowSize:
            ideal_action_num -= is_ideal_actions.popleft()

        success = grid.step(action)
        reward = 1.0 if success else -0.1

        next_state = grid.state()
        next_state = torch.tensor(next_state, dtype=torch.float32)
        _, next_value = network(next_state.unsqueeze(0))
        td = reward + kGamma * next_value[0].item() - value
        value_loss = td * td

        log_prob = torch.log_softmax(policy_logit, 0)[action]
        actor_loss = -log_prob * td.detach()

        loss = actor_loss + 0.1 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"i = {i:6d}, action = {action}, reward = {reward:.1f} value = {value.item():.4f} td = {td.item():.4f} ideal_action_rate = {100.0 * ideal_action_num / kWindowSize} is_ideal = {is_ideal_action}")
        f.write(
            f"{i}\t{value_loss.item():.4f}\t{actor_loss.item():.4f}\t{success}\t{is_ideal_action}\n")


if __name__ == "__main__":
    main()
