import argparse

import numpy as np

from datamodules.dtw import DTW
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def visualize(dataset_name):
    global data_dir, axes
    data_dir = f"data/{dataset_name}"
    print("Loading train data")
    dtw_train, labels_train, STS_train = DTW.load_data(f"{data_dir}/DTWs_train.npz")
    print(f"Train shape {dtw_train.shape}")
    print("Loading test data")
    dtw_test, labels_test, STS_test = DTW.load_data(f"{data_dir}/DTWs_test.npz")
    print(f"Test shape {dtw_test.shape}")
    while True:
        while True:
            random_train_idx = np.random.choice(range(len(dtw_train)), 1)[0]
            random_test_idx = np.random.choice(range(len(dtw_test)), 1)[0]

            random_label = np.random.choice(dtw_train.shape[-1], 1)[0]
            train_label_idx = np.argwhere(labels_train[random_train_idx] == random_label)
            test_label_idx = np.argwhere(labels_test[random_test_idx] == random_label)

            train_label_idx = train_label_idx[dtw_train.shape[1]: dtw_train.shape[1] * 2]
            test_label_idx = test_label_idx[dtw_train.shape[1]: dtw_train.shape[1]*2]

            if len(test_label_idx) > 0 and len(train_label_idx) > 0:
                break

        num_multi = STS_train[random_train_idx].shape[1]
        fig, axes = plt.subplots(num_multi + 1, 2, figsize=(12, 4))

        axes[0][0].imshow(
            dtw_train[random_train_idx, :, train_label_idx[0][0]:train_label_idx[-1][0] + 1, random_label],
            aspect='auto', origin='lower')
        axes[0][0].grid(True)
        axes[0][0].set_title(f'Train label:{random_label}')

        axes[0][1].imshow(dtw_test[random_test_idx, :, test_label_idx[0][0]:test_label_idx[-1][0] + 1, random_label],
                          aspect='auto', origin='lower')
        axes[0][1].grid(True)
        axes[0][1].set_title(f'Test label:{random_label}')

        for i in range(num_multi):
            axes[i + 1][0].plot(STS_train[random_train_idx, train_label_idx[0][0]:train_label_idx[-1][0] + 1, i])
            axes[i + 1][0].set_xticks([])
            axes[i + 1][0].set_xlim([0, dtw_train.shape[1] - 1])
            axes[i + 1][0].grid(True)

            axes[i + 1][1].plot(STS_test[random_test_idx, test_label_idx[0][0]:test_label_idx[-1][0] + 1, i])
            axes[i + 1][1].set_xticks([])
            axes[i + 1][1].set_xlim([0, dtw_train.shape[1] - 1])
            axes[i + 1][1].grid(True)

        # fig.tight_layout()
        plt.show()

        print("Enter (C) for continue and (B) for stopping")
        val = input(">>>:")
        if val.lower() == 'c':
            plt.close()
            continue
        if val.lower() == 'b':
            plt.close()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the data from the UCR_UEA database')
    parser.add_argument('--dataset_name', type=str,
                        help='Name of the dataset to be visualized after been created using create_DTWs')
    args = parser.parse_args()
    visualize(dataset_name=args.dataset_name)
