from collections import Counter

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


class DTW(Dataset):

    def __init__(self, dtws, labels, window_size=5, transform=None, target_transform=None):

        self.labels = labels
        self.dtws = dtws
        self.transform = transform
        self.target_transform = target_transform
        self.window_size = window_size

    @staticmethod
    def load_data(path):
        with np.load(path, 'r') as data:
            labels = data['STS_labels']
            dtws = data['DTWs']
            STS = data['STS']

        return dtws, labels, STS

    def __len__(self):
        return (self.labels.shape[1] + 1 - self.window_size) * self.dtws.shape[0]

    def __getitem__(self, idx):

        dtw_idx = idx % self.dtws.shape[0]
        windows_idx = idx // self.dtws.shape[0]

        labels = self.labels[dtw_idx, windows_idx:windows_idx + self.window_size]
        label_counts = dict(Counter(labels))
        label = int(max(label_counts, key=label_counts.get))
        dtw_window = self.dtws[dtw_idx, :, windows_idx:windows_idx + self.window_size, :]

        if self.transform:
            dtw_window = self.transform(dtw_window)
        if self.target_transform:
            label = self.target_transform(label)

        return dtw_window, label


class dtwDataModule(pl.LightningDataModule):

    def __init__(self, npz_path: str, batch_size: int = 32, num_workers: int = 4):
        print("Preparing data")
        super().__init__()
        self.data_dir = npz_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        print("Done")

    def prepare_data(self, window_size=15) -> None:
        print("Preparing data")

        print("Loading train data")
        dtw_train, labels_train, _ = DTW.load_data(f"{self.data_dir}/DTWs_train.npz")
        print(f"Train shape {dtw_train.shape}")

        print("Loading test data")
        dtw_test, labels_test, _ = DTW.load_data(f"{self.data_dir}/DTWs_test.npz")
        print(f"Test shape {dtw_test.shape}")

        val_idxs, test_idxs = train_test_split(range(len(dtw_test)), train_size=0.5, test_size=0.5)

        dtw_val, labels_val = dtw_test[val_idxs], labels_test[val_idxs]
        dtw_test, labels_test = dtw_test[test_idxs], labels_test[test_idxs]

        print(f"Train size {dtw_train.shape[0]}, val size {dtw_val.shape[0]}, test size {dtw_test.shape[0]}")

        avg, std = np.average(dtw_train), np.std(dtw_train)
        transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((avg,), (std,))])

        self.dtw_train = DTW(dtws=dtw_train,
                             labels=labels_train,
                             window_size=window_size,
                             transform=transform)

        print("DTW TRAIN COMPLETED")

        self.dtw_train.transform = transform

        self.dtw_val = DTW(dtws=dtw_val,
                           labels=labels_val,
                           window_size=window_size,
                           transform=transform)

        print("DTW VAL COMPLETED")

        self.dtw_test = DTW(dtws=dtw_test,
                            labels=labels_test,
                            window_size=window_size,
                            transform=transform)

        print("DTW TEST COMPLETED")

        self.channels = self.channels = np.unique(self.dtw_train.labels).shape[0]

        print("Done")

    def train_dataloader(self):
        return DataLoader(self.dtw_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dtw_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dtw_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dtw_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
