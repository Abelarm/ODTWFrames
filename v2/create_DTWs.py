import argparse
import multiprocessing
import os
from functools import partial
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('v1'))
sys.path.insert(0, os.path.abspath('time_series_augmentation'))

from v1.dataset.generate_files.utils.distance import compute_odtw_distance_matrix
import time_series_augmentation.utils.augmentation as aug


def wrapper_compute(x, lock, medoids, dataset_shape, num_reference, STS, real_rho):
    total_iteration = medoids.shape[0]
    with lock:
        bar = tqdm(
            desc=f'STS {x}',
            total=total_iteration,
            position=x,
            leave=True
        )

    partial_DTWs = np.zeros((medoids.shape[1], dataset_shape[1] * num_reference, medoids.shape[0]))
    for medoid_idx in range(total_iteration):
        if STS[x].shape[-1] > 1:
            partial_DTWs[:, :, medoid_idx] = compute_odtw_distance_matrix(medoids[medoid_idx], STS[x], real_rho,
                                                                          dist="euclidean_new")
        else:
            partial_DTWs[:, :, medoid_idx] = compute_odtw_distance_matrix(medoids[medoid_idx], STS[x], real_rho,
                                                                          dist="euclidean")
        with lock:
            bar.update(1)

    with lock:
        bar.close()
    return partial_DTWs


aug_probability = {
    'jitter': 0,
    'scaling': 0,
    'time_warp': 0,
    'window_warp': 0,
}


def create_dtws(dataset_name, rho, num_reference, tot_STS, num_process):
    dataset_path = os.path.join("data", dataset_name, "DB.npz")
    with np.load(dataset_path, allow_pickle=True) as data:
        X_train = data['X_train']
        Y_train = data['Y_train']

        X_test = data['X_test']
        Y_test = data['Y_test']

        train_size = data['train_size']
        test_size = data['test_size']

        medoids = data['medoids']
        medoids_idxs = data['medoids_idxs']

        if 'mapping' in data:
            mapping = eval(str(data['mapping']))
        else:
            mapping = None
    print(f"Medoids shapes {medoids.shape}")
    train_STS = int(tot_STS * train_size)
    test_STS = tot_STS - train_STS

    if '0' in Y_train and '0' in Y_test:
        zero_min = True
    else:
        zero_min = False

    for type in ['train', 'test']:
        print(f"Computing DTWs for {type}")
        if type == 'train':
            dataset = X_train
            labels = Y_train
            dataset_shape = X_train.shape
            num_STS = train_STS
        else:
            dataset = X_test
            labels = Y_test
            dataset_shape = X_test.shape
            num_STS = test_STS

        STS = np.empty((num_STS, dataset_shape[1] * num_reference, dataset_shape[2]))
        print(f"STS size {STS.shape}")
        STS_labels = np.empty((num_STS, dataset_shape[1] * num_reference))

        aug_shape = (1, dataset_shape[1], dataset_shape[2])
        for sts_idx in range(num_STS):

            for i in range(num_reference):
                while True:
                    random_idx = np.random.randint(0, dataset_shape[0])
                    if random_idx in medoids_idxs and type == 'train':
                        continue
                    else:
                        break

                if type == 'train':
                    selected_slice = dataset[random_idx].reshape(aug_shape)
                    if np.random.random() <= aug_probability['jitter']:
                        selected_slice = aug.jitter(selected_slice)
                    if np.random.random() <= aug_probability['scaling']:
                        selected_slice = aug.scaling(selected_slice)
                    if np.random.random() <= aug_probability['time_warp']:
                        selected_slice = aug.time_warp(selected_slice)
                    if np.random.random() <= aug_probability['window_warp']:
                        selected_slice = aug.window_warp(selected_slice)
                    selected_slice = selected_slice[0]
                else:
                    selected_slice = dataset[random_idx]

                STS[sts_idx, i * dataset_shape[1]:(i + 1) * dataset_shape[1]] = selected_slice
                if mapping:
                    single_label = [mapping[labels[random_idx]]]
                else:
                    tmp_label = int(float(labels[random_idx]))
                    if zero_min:
                        single_label = [tmp_label]
                    else:
                        single_label = [tmp_label - 1]


                STS_labels[sts_idx, i * dataset_shape[1]:(i + 1) * dataset_shape[1]] = np.array(
                    single_label * dataset_shape[1])

        save_path = os.path.join("data", dataset_name)
        os.makedirs(save_path, exist_ok=True)

        full_DTWs = np.zeros((num_STS, medoids.shape[1], dataset_shape[1] * num_reference, medoids.shape[0]))

        real_rho = rho ** (1 / medoids.shape[1])

        tot = []
        for sts_idx in range(num_STS):
            tot.append(sts_idx)

        lock = multiprocessing.Manager().Lock()

        wrapper_compute_call = partial(wrapper_compute,
                                       lock=lock,
                                       medoids=medoids,
                                       dataset_shape=dataset_shape,
                                       num_reference=num_reference,
                                       STS=STS,
                                       real_rho=real_rho)

        with Pool(processes=num_process) as pool:
            full_DTWs = pool.map(wrapper_compute_call, tot)

        full_DTWs = np.array(full_DTWs)
        full_DTWs = full_DTWs[:, :, 10:, :]
        print("Computation DONE")
        print("Saving the DTWs")
        np.savez_compressed(os.path.join(save_path, f'DTWs_{type}.npz'), STS=STS[:, 10:], STS_labels=STS_labels[:, 10:],
                            DTWs=full_DTWs)
        print("Saving DONE")
    plt.imshow(full_DTWs[0, :, 10:, 0], aspect='auto', origin='lower')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the DTWs from the downloaded dataset')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--rho', type=float, default=0.1,
                        help='Value of the rho')

    parser.add_argument('--num_reference', type=int, default=55,
                        help='Number of reference inside each Streaming Times Series')

    parser.add_argument('--tot_sts', type=int, default=25,
                        help='Total number Streaming Times Series to create')

    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of epochs to train the network')

    args = parser.parse_args()

    create_dtws(dataset_name=args.dataset_name, rho=args.rho,
                num_reference=args.num_reference, tot_STS=args.tot_sts, num_process=args.num_workers)
