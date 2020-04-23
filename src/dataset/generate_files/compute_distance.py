"""
Compute distance matrix for experimentation.
13-11-2019 :: Izaskun Oregui
"""

import numpy as np
import os

from dataset.generate_files.DataGenerator.utils import compute_classpercentages
from dataset.generate_files.OEMbatch import OEM_batch


def refMedoids(dataPath):
    """
    Finds medoid time series in data.
    Arguments:
        dataPath {string} -- path of the data file
    """

    # load data
    data = np.load(dataPath)
    labels = data[:, 0]
    data = data[:, 1:]
    numclass = np.unique(labels)

    medoidIDX = np.zeros((len(numclass),), dtype=int)
    aux = 0
    for label in numclass:
        # compute dtw among time series.
        tsIdx = np.where(labels == label)[0]
        dataSubset = data[tsIdx, :]
        distMAt = np.zeros((len(tsIdx), len(tsIdx)))
        for row in range(len(tsIdx)):
            for col in range(row + 1, len(tsIdx)):
                distMAt[row, col] = _dtw(
                    dataSubset[row, :], dataSubset[col, :])
        # find medoid
        medoid = np.argmin(np.sum(distMAt.T + distMAt, axis=0))
        medoidIDX[aux] = tsIdx[medoid]
        aux += 1
    return medoidIDX


def _dtw(x, y):
    """
    dtw between x and y time series

    Arguments:
        x, y {1-d array} -- time series

    Returns:
        float -- dtw between x and y
    """

    Y, X = np.meshgrid(y, x)

    RS = np.sqrt((X - Y) ** 2)
    r, s = np.shape(RS)

    # Solve first row
    for j in range(1, s):
        RS[0, j] += RS[0, j - 1]

    # Solve first column
    for i in range(1, r):
        RS[i, 0] += RS[i - 1, 0]

    # Solve the rest
    for i in range(1, r):
        for j in range(1, s):
            RS[i, j] += np.min([RS[i - 1, j], RS[i - 1, j - 1], RS[i, j - 1]])

    return RS[-1, -1]


def compute_odtw_distance_matrix(ref, stream, rho):
    """
    Compute distance matrix.
    :param refmat: reference pattern
    :param stream: stream time series
    :param rho: memory
    :return: distance matrices
    """

    print('Computing ODTW distance matrix')

    distMat = np.zeros((ref.size, stream.size))
    odtw = OEM_batch(ref, rho)
    distMat[:, :3] = odtw.init_dist(stream[:3])

    for point in range(3, stream.size):
        distMat[:, point] = odtw.update_dist([stream[point]])

    return distMat


# ! ------------------------------------------------------------------------------------------- GLOBAL PARAMETERS
pattern = 'cbf'  # database name
sub_pattern = True
length = 100  # length of the reference patterns
noise_level = 5  # std white noise (rate)
warp_level, shift_level = 10, 10
cycles_in_stream = 10  # number of patterns per label in the stream
patterns_in_ref = 10  # number of  pattern per label in the reference set
rho = 0.100  # memory parameter

# stream type
sets_list = ['train', 'validation', 'test']  # streaming sets
num_streams_set = [20, 5, 10]  # number of streams in each set

# *                                                     data folders and files
core_path = '../../../data'
folder_name = f'/{pattern}'
sub_folder = f'/rho {rho:.3f}'
if sub_pattern:
    sub_folder += '_base'

if pattern == 'gunpoint':
    num = 5
    length = 150
    fileREF = f'/REF_num-{num}.npy'
    fileSTREAM = f'/STREAM_cycles-per-label-{cycles_in_stream}'
else:
    fileSTREAM = '/STREAM_length-%d_noise-%d_warp-%d_shift-%d_outliers-0_cycles-per-label-%d' % \
                 (length, noise_level, warp_level, shift_level, cycles_in_stream)
    fileREF = '/REF_length-%d_noise-%d_warp-%d_shift-%d_outliers-0_num-%d.npy' % \
              (length, noise_level, warp_level, shift_level, patterns_in_ref)

if sub_pattern:
    fileREF = f'/BASE_REF_len-{length}_noise-5_num-1.npy'

    # *                                                     odtw core file names

    # these folders  finish: _ref-id-%d_stream-id-%d.npy
    fileODTWtrain = '/dtwMat-train_length-%d_noise-%d_warp-%d_shift-%d_outliers-0_rho-%.3f' % \
                    (length, noise_level, warp_level, shift_level, rho)
    fileODTWval = '/dtwMat-validation_length-%d_noise-%d_warp-%d_shift-%d_outliers-0_rho-%.3f' % \
                  (length, noise_level, warp_level, shift_level, rho)
    fileODTWtest = '/dtwMat-test_length-%d_noise-%d_warp-%d_shift-%d_outliers-0_rho-%.3f' % \
                   (length, noise_level, warp_level, shift_level, rho)

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

if not os.path.isdir(core_path + folder_name + sub_folder):
    print('creating directory : ' + core_path + folder_name + sub_folder)
    os.makedirs(core_path + folder_name + sub_folder)

# ! --------------------------------------------------------------------------------------------- COMPUTE DTW

# ------------------------------- reference patterns load data
REF = np.load(core_path + folder_name + fileREF)
labelsREF = REF[:, 0]
REF = REF[:, 1:]

for s in range(len(sets_list)):
    print('SET :: %s' % sets_list[s])
    if sets_list[s] in ['validation', 'train', 'test']:
        refIDs = refMedoids(core_path + folder_name + fileREF)
    else:
        refIDs = np.arange(REF.shape[0], dtype=int)
    for j in range(num_streams_set[s]):
        # ------------------------------- load streams
        if sets_list[s] == 'test' and pattern == 'gunpoint':
            cycles_in_stream_test = 20
            fileSTREAM = f'/STREAM_cycles-per-label-{cycles_in_stream_test}'
        file = '%s_set-%s_id-%d.npy' % (fileSTREAM, sets_list[s], j)
        STREAM = np.load(core_path + folder_name + file)
        labelsSTREAM = STREAM[:, 1]
        STREAM = STREAM[:, 0]
        print(np.shape(STREAM))

        for r in refIDs:
            if pattern in ['cbf', 'two_patterns2', 'two_patterns', 'rational', 'synthetic_control']:
                fileODTW = sub_folder + '/dtwMat-%s_length-%d_noise-%d_warp-%d_shift-%d' \
                                        '_outliers-0_rho-%.3f_ref-id-%d_stream-id-%d.npy' \
                           % (sets_list[s], length, noise_level, warp_level, shift_level, rho, r, j)
            else:
                fileODTW = sub_folder + '/dtwMat-%s_rho-%.3f_ref-id-%d_stream-id-%d.npy' \
                           % (sets_list[s], rho, r, j)
            if os.path.isfile(core_path + folder_name + fileODTW) is False:
                print('Computing ODTW between (ref, stream_id) = (%d, %d)' % (r, j))
                distMat = compute_odtw_distance_matrix(
                    REF[r, :], STREAM, rho ** (1.0 / length))
                np.save(core_path + folder_name + fileODTW, distMat)
            else:
                print('ODTW between (ref, stream_id) = (%d, %d) COMPUTED' % (r, j))
