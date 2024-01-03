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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        feature_dim = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.positional_encoding = PositionalEncoding(feature_dim, 20)
        transformer_layer = nn.TransformerEncoderLayer(
            feature_dim, 8, dim_feedforward=(feature_dim * 4), dropout=0.0, norm_first=True)
        layers = nn.TransformerEncoder(transformer_layer, 1)
        liner = nn.Linear(feature_dim, out_channels)
        self.transformer = nn.Sequential(layers, liner)

    def forward(self, x):
        bs, seq_len, c, h, w = x.shape
        x = x.view((bs * seq_len, c, h, w))
        x = self.encoder(x)
        x = x.view((bs, seq_len, -1))
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x[:, -1]
        return x


if __name__ == "__main__":
    bs = 32
    seq_len = 20
    ch = 2
    h = 36
    w = 36
    out_ch = 5

    data = torch.randn((bs, seq_len, ch, h, w))
    print(data.shape)

    model = QNetwork(ch, 5)
    out = model(data)
    print(out.shape)

    model = TransformerQNetwork(ch, 5)
    out = model(data)
    print(out.shape)
