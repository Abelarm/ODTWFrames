import torch
from pytorch_lightning import LightningModule
from torch import nn


class RNN_TS(LightningModule):

    def __init__(self, ref_size, channels, window_size):
        super().__init__()

        self.channels = channels
        self.window_size = window_size
        self.n_feature_maps = 32

        self.lstm_1 = nn.LSTM(self.channels, hidden_size=self.n_feature_maps, dropout=0.2, batch_first=True)
        self.bn_1 = nn.BatchNorm1d(num_features=self.n_feature_maps, momentum=0.999, eps=0.01)
        self.lstm_2 = nn.LSTM(input_size=self.n_feature_maps, hidden_size=self.n_feature_maps * 2, dropout=0.2, batch_first=True)
        self.bn_2 = nn.BatchNorm1d(num_features=self.n_feature_maps * 2, momentum=0.999, eps=0.01)
        self.lin = nn.Linear(in_features=self.n_feature_maps * 2 * window_size,
                             out_features=self.n_feature_maps * 4 * window_size)
        self.relu = nn.ReLU()

    def get_output_shape(self):
        return self.n_feature_maps * 4 * self.window_size

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1)).float()

        hidden = (torch.randn(1, x.shape[0], self.n_feature_maps).cuda(),
                  torch.randn(1, x.shape[0], self.n_feature_maps).cuda())

        output, _ = self.lstm_1(x, hidden)
        ret_bn1 = self.bn_1(output.permute(0, 2, 1))

        hidden_2 = (torch.randn(1, x.shape[0], self.n_feature_maps*2).cuda(),
                    torch.randn(1, x.shape[0], self.n_feature_maps*2).cuda())

        output, (h_t2, _) = self.lstm_2(ret_bn1.permute(0, 2, 1), hidden_2)
        ret_bn2 = self.bn_2(output.permute(0, 2, 1))
        ret_lin = self.lin(ret_bn2.view(ret_bn2.shape[0], -1))
        return self.relu(ret_lin)
