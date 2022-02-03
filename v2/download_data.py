import argparse

import numpy as np
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix
import os
import matplotlib.pyplot as plt


def download_dataset(dataset_name):
    X, Y = load_UCR_UEA_dataset(name=dataset_name,
                                return_X_y=True)
    train_size = 0.9
    test_size = 0.1
    try:
        float(Y[0])
        mapping = None
    except ValueError:
        mapping = {k: v for v, k in enumerate(np.unique(Y))}
    shapes = X.shape
    shapes = shapes + X.iloc[0][X.iloc[0].index[0]].shape
    print(f"Sample shape {shapes}")

    print(f"Num classes {len(np.unique(Y))}")
    while True:
        train_idx, test_idx = train_test_split(range(shapes[0]), train_size=train_size, test_size=test_size)
        y_train, y_test = Y[train_idx], Y[test_idx]

        if len(np.unique(y_train)) == len(np.unique(y_test)) and all(np.unique(y_train) == np.unique(y_test)):
            break
    full_array_train = np.zeros((len(train_idx), shapes[2], shapes[1]))
    Y_train = []
    full_array_test = np.zeros((len(test_idx), shapes[2], shapes[1]))
    Y_test = []
    for j, i in enumerate(train_idx):
        sub_array = np.zeros(shapes[1:])
        for cnt, index in enumerate(X.iloc[i].index):
            sub_array[cnt, :] = X.iloc[i][index].values
        full_array_train[j, :, :] = sub_array.transpose()
        Y_train.append(Y[i])
    Y_train = np.array(Y_train)
    for j, i in enumerate(test_idx):
        sub_array = np.zeros(shapes[1:])
        for cnt, index in enumerate(X.iloc[i].index):
            sub_array[cnt, :] = X.iloc[i][index].values
        full_array_test[j, :, :] = sub_array.transpose()
        Y_test.append(Y[i])
    Y_test = np.array(Y_test)
    medoids = np.zeros((len(np.unique(Y_train)), shapes[2], shapes[1]))
    medoids_idxs = np.zeros(len(np.unique(Y_train)))
    for i, y in enumerate(np.unique(Y_train)):
        index = np.argwhere(Y_train == y)
        sub_array = full_array_train[index.reshape(-1)]

        shapes = sub_array.shape
        new_shapes = (shapes[0], shapes[1] * shapes[2])
        medoid_idx = np.argmin(distance_matrix(sub_array.reshape(new_shapes), sub_array.reshape(new_shapes)).sum(0))
        print(f'For class {y} medoid_idx: {index[medoid_idx]}')

        medoids[i] = sub_array[medoid_idx]
        medoids_idxs[i] = index[medoid_idx]
    save_path = os.path.join("data", dataset_name)
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, 'DB.npz'),
                        medoids=medoids, medoids_idxs=medoids_idxs,
                        X_train=full_array_train, Y_train=Y_train,
                        X_test=full_array_test, Y_test=Y_test,
                        train_size=train_size, test_size=test_size,
                        mapping=mapping)
    while True:

        if not mapping:
            random_label = np.random.choice(np.unique(Y_train), 1)[0]
        else:
            random_label = np.random.choice(len(mapping), 1)[0]
        tmp_Y_train = [int(float(i)) if not mapping else mapping[i] for i in Y_train]
        tmp_Y_test = [int(float(i)) if not mapping else mapping[i] for i in Y_test]

        train_label_idx = np.argwhere(np.array(tmp_Y_train) == int(float(random_label)))[0]
        test_label_idx = np.argwhere(np.array(tmp_Y_test) == int(float(random_label)))[0]

        if len(test_label_idx) > 0 and len(train_label_idx) > 0:
            break
    num_multi = full_array_train.shape[2]
    fig, axes = plt.subplots(num_multi, 2, figsize=(12, 4))

    if num_multi > 1:
        for i in range(num_multi):
            axes[i][0].plot(full_array_train[train_label_idx[0], :, i])
            axes[i][0].set_xticks([0, full_array_train.shape[1] - 1])
            axes[i][0].set_xlim([0, full_array_train.shape[1] - 1])
            axes[i][0].grid(True)
            axes[i][0].set_title('STS_train')

            axes[i][1].plot(full_array_test[test_label_idx[0], :, i])
            axes[i][1].set_xticks([0, full_array_train.shape[1] - 1])
            axes[i][1].set_xlim([0, full_array_train.shape[1] - 1])
            axes[i][1].grid(True)
            axes[i][1].set_title('STS_test')
    else:
        axes[0].plot(full_array_train[train_label_idx[0]])
        axes[0].set_xticks([0, full_array_train.shape[1] - 1])
        axes[0].set_xlim([0, full_array_train.shape[1] - 1])
        axes[0].grid(True)
        axes[0].set_title('STS_train')

        axes[1].plot(full_array_test[test_label_idx[0]])
        axes[1].set_xticks([0, full_array_train.shape[1] - 1])
        axes[1].set_xlim([0, full_array_train.shape[1] - 1])
        axes[1].grid(True)
        axes[1].set_title('STS_test')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the data from the UCR_UEA database')
    parser.add_argument('--dataset_name', type=str,
                        help='Name of the dataset to be download')
    args = parser.parse_args()
    download_dataset(dataset_name=args.dataset_name)
