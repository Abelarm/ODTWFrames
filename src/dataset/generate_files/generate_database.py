"""
2019-10-28 : Izaskun Oregui

pattern must be one of the followings.
    1) arma              ----> 8 classes
    2) synthetic_control ----> 6 classes
    3) sines             ----> 5 classes
    4) kohlerlorenz      ----> 5 classes
    5) cbf               ----> 3 classes
    6) two_patterns      ----> 4 classes
    7) rational          ----> 4 classes
    8) seasonal          ----> 4 classes

"""

import numpy as np
import os
import sys

from dataset.generate_files.DataGenerator.sts_generator import sts_generator
from dataset.generate_files.DataGenerator.utils import compute_classpercentages
from dataset.generate_files.utils import general


def resort_label_array(label_array, seed):
    """
    resort labels_array so the resulting array do not have two consecutive patters with same class label.
    :param label_array: array of labels
    :return:  the indices that sorts label_array
    """

    np.random.seed(seed)

    # ------------------------------- parameters
    NUM = len(label_array)
    labelsStreamAux = label_array.copy()
    class_labels, counts = np.unique(label_array, return_counts=True)
    rearrange = np.zeros((NUM,), dtype=int)
    indices = np.arange(NUM, dtype=int)

    # ------------------------------- resort
    label = None
    for i in range(NUM):
        stop = 0
        if i == 0:
            label = np.random.choice(class_labels, 1)
            counts[class_labels == label] -= 1
            rearrange[i], labelsStreamAux, indices = general.delete_element(
                labelsStreamAux, label, indices, seed_stream + (i + 1))
        else:
            while stop == 0:
                aux = np.random.choice(
                    class_labels, 1, p=counts / len(labelsStreamAux))
                noneZeros = np.count_nonzero(counts)
                if aux != label or noneZeros == 1:
                    stop = 1
                    label = aux.copy()
                    counts[class_labels == label] -= 1
                    rearrange[i], labelsStreamAux, indices = general.delete_element(labelsStreamAux, label, indices,
                                                                                    seed_stream + (i + 1))
    return rearrange


# ! ------------------------------------------------------------------------------------------- GLOBAL PARAMETERS
pattern = 'rational'  # database name
length = 100  # length of the reference patterns
noise_level = 5  # std white noise (rate)
warp_level, shift_level = 10, 10
cycles_in_stream = 10  # number of patterns per label in the stream
patterns_in_ref = 10  # number of  pattern per label in the reference set

# stream type
sets_list = ['train', 'validation', 'test']  # streaming sets
num_streams_set = [25, 10, 15]  # number of streams in each set

# *                                                     saving folders and files
core_path = '../../../data'
folder_name = '/' + pattern
if os.path.isdir(core_path + folder_name) is False:
    print('Generate ' + core_path + folder_name + ' folder')
    os.makedirs(core_path + folder_name)

fileREF = '/REF_length-%d_noise-%d_warp-%d_shift-%d_outliers-0_num-%d.npy' % \
    (length, noise_level, warp_level, shift_level, patterns_in_ref)
fileSTREAM = '/STREAM_length-%d_noise-%d_warp-%d_shift-%d_outliers-0_cycles-per-label-%d' % \
    (length, noise_level, warp_level, shift_level, cycles_in_stream)


# ! --------------------------------------------------------------------------------------------- INITIALIZATION
classpercentages = compute_classpercentages(pattern)
numLabels = len(classpercentages)
num_stream_cycles = cycles_in_stream * numLabels
num_ref_patterns = patterns_in_ref * numLabels

seed_ref = 123 + sum([ord(char) for char in pattern])
seed_stream = 456 + sum([ord(char) for char in pattern])
if not os.path.isdir(core_path + folder_name):
    print('creating directory : ' + core_path + folder_name)
    os.makedirs(core_path + folder_name)

# ! ------------------------------------------------------------------------------------------------- SIMULATE DATABASES

#  * -------------------------------------------------------- reference patters
if os.path.isfile(core_path + folder_name + fileREF) is False:
    genRef = sts_generator(
        num_series=num_ref_patterns, seed=seed_ref)
    dataRef, labelsRef = genRef.generate(
        name=pattern,
        length_series=length,
        noise_level=noise_level,
        outlier_level=0,
        shift_level=shift_level,
        classpercentages=classpercentages,
        warp_level=warp_level)
    dataREF = general.Znorm(dataRef)
    REF = np.column_stack((labelsRef, dataRef))
    print('saving : ' + core_path + folder_name + fileREF)
    np.save(core_path + folder_name + fileREF, REF)
else:
    print('******* WARNING :: Ref time series already computed')
    sys.exit()

# ------------------------------- stream time series (for train test and val)
for s in range(len(sets_list)):
    for j in range(num_streams_set[s]):
        # ------------------------------- generate patterns
        genStream = sts_generator(
            num_series=num_stream_cycles,
            seed=seed_stream + sum([ord(char) for char in sets_list[s]]) + j + 1)
        dataStream, labelsStream = genStream.generate(
            name=pattern,
            length_series=length,
            noise_level=noise_level,
            outlier_level=0,
            shift_level=shift_level,
            classpercentages=classpercentages,
            warp_level=warp_level)
        dataStream = general.Znorm(dataStream)

        # ------------------------------- rearrange labels
        rearrange = resort_label_array(
            labelsStream, seed=seed_stream + j + sum([ord(char) for char in sets_list[s]]))
        labelsStream = labelsStream[rearrange]
        dataStream = dataStream[rearrange, :]
        STREAM = np.zeros((dataStream.size, 2))
        for i in range(dataStream.shape[0]):
            ini = i * length
            fin = (i + 1) * length
            STREAM[ini:fin, 0] = dataStream[i, :]
            STREAM[ini:fin, 1] = labelsStream[i]

        # ------------------------------- save label
        file = '%s_set-%s_id-%d.npy' % (fileSTREAM, sets_list[s], j)
        print('saving : ' + core_path + folder_name + file)
        np.save(core_path + folder_name + file, STREAM)
