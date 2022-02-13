import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F


class ResidualBlock_1d(LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(out_channels),
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm1d(out_channels)
        )

        self.block.apply(ResidualBlock_1d.initialize_weights)
        self.shortcut.apply(ResidualBlock_1d.initialize_weights)

    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
        elif hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        block = self.block(x.float())
        shortcut = self.shortcut(x.float())

        block = torch.add(block, shortcut)
        return F.relu(block)


class ResNet_TS(LightningModule):

    def __init__(self, ref_size, channels, window_size):
        super().__init__()

        self.channels = channels
        self.n_feature_maps = 32

        self.model = nn.Sequential(
            ResidualBlock_1d(in_channels=channels, out_channels=self.n_feature_maps),
            ResidualBlock_1d(in_channels=self.n_feature_maps, out_channels=self.n_feature_maps * 2),
            ResidualBlock_1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 2),
            ResidualBlock_1d(in_channels=self.n_feature_maps * 2, out_channels=self.n_feature_maps * 4),
            nn.AvgPool1d(window_size)
        )

    def get_output_shape(self):
        return (len(self.model._modules) - 1) * self.n_feature_maps

    def forward(self, x):
        return self.model(x.float())
