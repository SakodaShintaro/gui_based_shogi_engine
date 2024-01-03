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
        self.encoder = nn.Sequential(
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
        self.transformer = nn.Sequential(layers, liner)

    def forward(self, x):
        bs, seq_len, c, h, w = x.shape
        x = x.view((bs * seq_len, c, h, w))
        x = self.encoder(x)
        x = x.view((bs, seq_len, -1))
        x = self.transformer(x)
        x = x[:, -1]
        return x


if __name__=="__main__":
    import torch
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
