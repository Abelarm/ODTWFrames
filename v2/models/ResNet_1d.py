import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


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

    def __init__(self, channels, labels, window_size, lr=0.0001):
        super().__init__()

        self.channels = channels
        self.lr = lr

        n_feature_maps = 32

        self.model = nn.Sequential(
            ResidualBlock_1d(in_channels=channels, out_channels=n_feature_maps),
            ResidualBlock_1d(in_channels=n_feature_maps, out_channels=n_feature_maps * 2),
            ResidualBlock_1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 2),
            ResidualBlock_1d(in_channels=n_feature_maps * 2, out_channels=n_feature_maps * 4),
            nn.AvgPool1d(window_size)
        )

        self.classifier = nn.Linear(in_features=128, out_features=labels)

        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1(num_classes=labels,
                                        average="micro")
        self.train_auroc = torchmetrics.AUROC(num_classes=labels,
                                              average="macro")

        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1(num_classes=labels,
                                      average="micro")
        self.val_auroc = torchmetrics.AUROC(num_classes=labels,
                                            average="macro")

        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1(num_classes=labels,
                                       average="micro")
        self.test_auroc = torchmetrics.AUROC(num_classes=labels,
                                             average="macro")

    def forward(self, x):
        feature = self.model(x.float())
        flat = feature.view(feature.size(0), -1)
        return self.classifier(flat)

    def _inner_step(self, x, y):
        logits = self(x)
        y_pred = logits.softmax(dim=-1)
        loss = F.cross_entropy(logits, y)
        return loss, y_pred

    def training_step(self, batch, batch_nb):

        x, y = batch
        loss, y_pred = self._inner_step(x, y)

        # accumulate and return metrics for logging
        acc = self.train_acc(y_pred, y)
        f1 = self.train_f1(y_pred, y)

        self.log("train_loss", loss)
        self.log("train_accuracy", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        # compute metrics
        train_accuracy = self.train_acc.compute()
        train_f1 = self.train_f1.compute()

        # log metrics
        self.log("epoch_train_accuracy", train_accuracy, prog_bar=True)
        self.log("epoch_train_f1", train_f1, prog_bar=True)

        # reset all metrics
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()
        print(f"\ntraining accuracy: {train_accuracy:.4}, " f"f1: {train_f1:.4}")

    def evaluate(self, batch, stage=None):
        x, y = batch
        loss, y_pred = self._inner_step(x, y)

        if stage == 'val':
            self.val_acc(y_pred, y)
            self.val_f1(y_pred, y)
            self.val_auroc(y_pred, y)

        elif stage == "test":
            self.test_acc(y_pred, y)
            self.test_f1(y_pred, y)
            self.test_auroc(y_pred, y)

        return loss

    def _custom_epoch_end(self, step_outputs, stage):

        if stage == "val":
            acc_metric = self.val_acc
            f1_metric = self.val_f1
            auroc_metric = self.val_auroc
        elif stage == "test":
            acc_metric = self.test_acc
            f1_metric = self.test_f1
            auroc_metric = self.test_auroc

        # compute metrics
        loss = torch.tensor(step_outputs).mean()
        accuracy = acc_metric.compute()
        f1 = f1_metric.compute()
        auroc = auroc_metric.compute()

        # log metrics
        self.log(f"{stage}_accuracy", accuracy)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_f1", f1)
        self.log(f"{stage}_auroc", auroc)

        # reset all metrics
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        print(f"\n{stage} accuracy: {accuracy:.4} " f"f1: {f1:.4}, auroc: {auroc:.4}")

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def validation_epoch_end(self, validation_step_outputs):
        self._custom_epoch_end(validation_step_outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def test_epoch_end(self, test_step_outputs):
        self._custom_epoch_end(test_step_outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=np.sqrt(0.1), patience=5, min_lr=0.5e-7),
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 10
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }