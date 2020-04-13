"""
frequently used functions
19-06-2019: Izaskun Oregi
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import zipfile


def Znorm(mat, channels=None):
    """
    Z-normalization of time series.
    mat: 2d or 3d array. Each row in mat represents a time series.
    channels: number of channels is mat. Default None. If channels is None mat is considered 2d.
    """

    if channels is None:
        mean = np.mean(mat, axis=1)
        std = np.std(mat, axis=1)
        for i in range(len(mean)):
            mat[i, :] = (mat[i, :] - mean[i]) / std[i]
    else:
        for c in range(channels):
            mean = np.mean(mat[:, :, c], axis=1)
            std = np.std(mat[:, :, c], axis=1)
            for i in range(len(mean)):
                mat[i, :, c] = (mat[i, :, c] - mean[i]) / std[i]

    return mat


def MaxMin(mat, channels=None):
    """
    MaxMin normalization of time series.
    mat: 2d or 3d array. Each row in mat represents a time series.
    channels: number of channels is mat. Default None. If channels is None mat is considered 2d.
    """

    if channels is None:
        max = np.max(mat)
        min = np.min(mat)
        mat = (mat - min) / (max - min)
    else:
        for c in range(channels):
            max = np.max(mat[:, :, c])
            min = np.min(mat[:, :, c])
            mat[:, :, c] = (mat[:, :, c] - min) / (max - min)
    return mat


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def arg_intersection(lst1, lst2):

    num_list1 = len(lst1)
    indices = [index for index in range(num_list1) if lst1[index] in lst2]
    return np.array(indices, dtype=int)


def complementary(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3


def delete_element(array, label, indices, seed):
    """
    delete element with class value, label, from array

    :param array: array of labels
    :param label: label to find
    :param indices: list of available indices
    :param seed: random seed
    :return: new_array with element deleted
    """

     # np.random.seed(seed=seed)
    idx_label = np.where(array == label)[0]
    np.random.shuffle(idx_label)

    return_index = indices[idx_label[0]]
    new_array = np.delete(array, idx_label[0])
    new_indices = np.delete(indices, idx_label[0])

    return return_index, new_array, new_indices


def dict2array(dict, axis=None):
    """
    Converts dictionary to array
    :param dict: dictionary
    :param axis: axis along which to stack dictionary elements
    :return: array
    """

    key_list = list(dict.keys())
    array = []
    for i in key_list:
        if i == key_list[0]:
            array = dict[i]
        else:
            array = np.concatenate((array, dict[i]), axis=axis)
    return array