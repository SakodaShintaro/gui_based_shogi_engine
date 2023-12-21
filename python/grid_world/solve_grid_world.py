import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from grid_world import GridWorld
from action import *
from replay_buffer import ReplayBuffer


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

    def forward(self, input_x):
        # x : [bs, 2, h, w]
        # NN推論する場合
        # x = input_x
        # for layer in self.conv_layers:
        #     x = layer(x)
        #     x = nn.ReLU()(x)
        # x = x.flatten(1)
        # policy = self.linear_policy(x)
        # value = self.linear_value(x)

        # テーブルベースで考える場合
        # 1になっているindexを取得する
        input_x = input_x.flatten(2)  # [bs, 2, h * w]
        dim0 = input_x[:, 0].nonzero(as_tuple=False)
        dim1 = input_x[:, 1].nonzero(as_tuple=False)
        dim0 = dim0[:, 1]
        dim1 = dim1[:, 1]
        index = dim0 * self.square + dim1
        policy = self.policy_embedding(index)
        value = self.value_embedding(index)

        return policy, value


def main():
    kGridSize = 4
    grid = GridWorld(kGridSize)

    device = torch.device("cpu")

    network = NeuralNetwork(kGridSize, kGridSize)
    network.to(device)

    optimizer = optim.SGD(network.parameters(), 1.0)

    buffer = ReplayBuffer(2)

    kGamma = 0.9  # 割引率

    is_ideal_actions = deque()
    kWindowSize = 200
    ideal_action_num = 0

    f = open("grid_world_log.tsv", "w")

    for i in range(20000):
        print("")
        # 実行フェーズ
        network.eval()
        grid.print()

        state = grid.state()
        state = torch.tensor(state, dtype=torch.float32)
        state = state.to(device)

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
        reward = 1.0 if success else -0.01

        buffer.push(state, action, reward, value.item())
        ideal_rate = 100.0 * ideal_action_num / kWindowSize
        print(f"i = {i:6d}, action = {action}, reward = {reward:+.1f} value = {value.item():+.4f} ideal_action_rate = {ideal_rate:5.1f} is_ideal = {is_ideal_action}")

        # 学習フェーズ
        network.train()
        samples = buffer.sample(4)
        if samples is None:
            continue

        states = []
        actions = []
        value_targets = []

        for sample in samples:
            states.append(sample[0].state)
            actions.append(sample[0].action)
            value_target = sample[-1].value
            for index in range(len(sample) - 2, -1, -1):
                value_target = kGamma * value_target + sample[index].reward
            value_targets.append(value_target)
        states = torch.stack(states).to(device)
        actions = torch.tensor(actions).to(device)
        value_targets = torch.tensor(value_targets).to(device)

        train_policies, train_values = network(states)
        td = value_targets - train_values
        value_loss = td * td
        value_loss = value_loss.mean()

        log_prob = torch.log_softmax(train_policies, 1)
        log_prob = log_prob.gather(1, actions.unsqueeze(1))
        policy_loss = -log_prob * td.detach()
        policy_loss = policy_loss.mean()

        loss = policy_loss + 0.1 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        after_policies, _ = network(states)
        train_policies = torch.softmax(train_policies, 1)
        after_policies = torch.softmax(after_policies, 1)
        for j in range(len(samples)):
            print(f"学習局面{j} : ", end="")
            print(f"reward = {samples[j][0].reward:+.2f}", end=", ")
            print(f"value_target = {value_targets[j].item():+.4f}", end=", ")
            print(f"value = {train_values[j].item():+.4f}", end=", ")
            print(f"td = {td[j].item():+.4f}")
            for h in range(kGridSize):
                for w in range(kGridSize):
                    v = 0
                    if samples[j][0].state[0][h][w] == 1:
                        v += 1
                    if samples[j][0].state[1][h][w] == 1:
                        v += 2
                    print(v, end="")
                print()

            for k in range(kActionSize):
                diff = after_policies[j][k] - train_policies[j][k]
                end = " <- curr_action\n" if k == actions[j] else "\n"
                print(
                    f"{train_policies[j][k]:+.4f} -> {after_policies[j][k]:+.4f} = {diff:+.4f}", end=end)

        f.write(
            f"{i}\t{value_loss.item():.4f}\t{policy_loss.item():.4f}\t{success}\t{is_ideal_action}\n")


if __name__ == "__main__":
    main()
