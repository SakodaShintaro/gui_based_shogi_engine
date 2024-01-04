import torch
from torch import nn
import math


class QNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
        )

    def forward(self, x):
        x = x[:, -1]
        x = self.network(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x


class TransformerQNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len):
        super().__init__()
        feature_dim = 64
        self.encoder_obs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.encoder_action = nn.Embedding(out_channels, feature_dim)

        self.encoder_reward = nn.Linear(1, feature_dim)

        self.positional_encoding = PositionalEncoding(feature_dim, seq_len * 3)
        transformer_layer = nn.TransformerEncoderLayer(
            feature_dim, 8, dim_feedforward=(feature_dim * 4), dropout=0.0, norm_first=True, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, 1)
        self.head = nn.Linear(feature_dim, out_channels)

    def forward(self, observation, action, reward):
        bs, seq_len, c, h, w = observation.shape
        observation = observation.view((bs * seq_len, c, h, w))
        observation = self.encoder_obs(observation)
        observation = observation.view((bs, seq_len, -1))

        action = self.encoder_action(action.squeeze(2))

        reward = self.encoder_reward(reward)

        x = torch.cat([observation, action, reward], dim=1)
        x = self.positional_encoding(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        x = self.transformer(x, mask=mask, is_causal=True)
        a = x[:, 0::3]
        r = x[:, 1::3]
        s = x[:, 2::3]
        a = self.head(a)
        return a
