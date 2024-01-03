from torch import nn


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


class TransformerQNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        feature_dim = 64
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        transformer_layer = nn.TransformerEncoderLayer(
            feature_dim, 8, dim_feedforward=(feature_dim * 4), dropout=0.0, norm_first=True)
        layers = nn.TransformerEncoder(transformer_layer, 1)
        liner = nn.Linear(feature_dim, out_channels)
        self.q_net = nn.Sequential(layers, liner)

    def forward(self, x):
        x = x[:, -1]
        x = self.network(x)
        return x
