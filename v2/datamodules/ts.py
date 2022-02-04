from collections import Counter


import pytorch_lightning as pl
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


class STS(Dataset):

    def __init__(self, sts, labels, window_size=5, transform=None, target_transform=None):

        self.labels = labels
        self.sts = sts
        self.transform = transform
        self.target_transform = target_transform
        self.window_size = window_size

    @staticmethod
    def load_data(path):
        with np.load(path, 'r') as data:
            labels = data['STS_labels']
            sts = data['STS']

        return sts, labels

    def __len__(self):
        return (self.labels.shape[1] + 1 - self.window_size) * self.sts.shape[0]

    def __getitem__(self, idx):

        sts_idx = idx % self.sts.shape[0]
        windows_idx = idx // self.sts.shape[0]

        labels = self.labels[sts_idx, windows_idx:windows_idx + self.window_size]
        label_counts = dict(Counter(labels))
        label = int(max(label_counts, key=label_counts.get))
        sts_window = self.sts[sts_idx, windows_idx:windows_idx + self.window_size, :]

        if self.transform:
            sts_window = self.transform(sts_window)
            sts_window = torch.squeeze(torch.permute(sts_window, (0, 2, 1)), dim=0)
        if self.target_transform:
            label = self.target_transform(label)

        return sts_window, label


class stsDataModule(pl.LightningDataModule):

    def __init__(self, npz_path: str, batch_size: int = 32, num_workers: int = 4, split=[0.9, 0.05, 0.05]):
        print("Preparing data")
        super().__init__()
        self.data_dir = npz_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        print("Done")

    def prepare_data(self, window_size=15) -> None:
        print("Preparing data")

        print("Loading train data")
        sts_train, labels_train = STS.load_data(f"{self.data_dir}/DTWs_train.npz")
        print(f"Train shape {sts_train.shape}")

        print("Loading test data")
        sts_test, labels_test = STS.load_data(f"{self.data_dir}/DTWs_test.npz")
        print(f"Test shape {sts_test.shape}")

        val_idxs, test_idxs = train_test_split(range(len(sts_test)), train_size=0.5, test_size=0.5)

        sts_val, labels_val = sts_test[val_idxs], labels_test[val_idxs]
        sts_test, labels_test = sts_test[test_idxs], labels_test[test_idxs]

        print(f"Train size {sts_train.shape[0]}, val size {sts_val.shape[0]}, test size {sts_test.shape[0]}")

        self.sts_train = STS(sts=sts_train, labels=labels_train, window_size=window_size)

        print("sts TRAIN COMPLETED")

        avg, std = np.average(self.sts_train.sts), np.std(self.sts_train.sts)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((avg,), (std,)),
                                        ])

        self.sts_train.transform = transform

        self.sts_val = STS(sts=sts_val,
                           labels=labels_val,
                           window_size=window_size,
                           transform=transform)

        print("STS VAL COMPLETED")

        self.sts_test = STS(sts=sts_test,
                            labels=labels_test,
                            window_size=window_size,
                            transform=transform)

        print("STS TEST COMPLETED")

        self.labels_size = np.unique(self.sts_train.labels).shape[0]

        print("Done")

    def train_dataloader(self):
        return DataLoader(self.sts_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.sts_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.sts_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.sts_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
