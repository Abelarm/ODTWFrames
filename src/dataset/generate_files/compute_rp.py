import os

import numpy as np
from scipy.spatial.distance import cdist

from dataset.generate_files.DataGenerator.utils import compute_classpercentages

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


# ! ------------------------------------------------------------------------------------------- GLOBAL PARAMETERS
pattern = 'rational'  # database name
dataset_type = 'RP'
sub_pattern = False
sub_pattern_names = ''
length = 100  # length of the reference patterns
noise_level = 5  # std white noise (rate)
warp_level, shift_level = 10, 10
cycles_in_stream = 10  # number of patterns per label in the stream
patterns_in_ref = 10  # number of  pattern per label in the reference set

# stream type
sets_list = ['train', 'validation', 'test']  # streaming sets
num_streams_set = [20, 5, 10]  # number of streams in each set

# *                                                     data folders and files
core_path = '../../../data'
folder_name = f'{pattern}'
sub_folder = dataset_type

if sub_pattern:
    sub_folder += '_base'
if len(sub_pattern_names) > 0:
    sub_folder += f'_{sub_pattern_names}'

if pattern == 'gunpoint':
    num = 5
    length = 150
    fileREF = f'REF_num-{num}.npy'
    fileSTREAM = f'STREAM_cycles-per-label-{cycles_in_stream}'
else:

    fileSTREAM = f'STREAM_length-{length}_noise-{noise_level}_warp-{warp_level}_shift-{shift_level}'\
                 f'_outliers-0_cycles-per-label-{cycles_in_stream}'

    fileREF = f'REF_length-{length}_noise-{noise_level}_warp-{warp_level}_shift-{shift_level}'\
              f'_outliers-0_num-{patterns_in_ref}.npy'


if sub_pattern:
    if len(sub_pattern_names) > 0:
        fileREF = f'BASE_REF_len-{length}_noise-5_num-1_{sub_pattern_names}.npy'
    else:
        fileREF = f'BASE_REF_len-{length}_noise-5_num-1.npy'


# ! --------------------------------------------------------------------------------------------- INITIALIZATION
classpercentages = compute_classpercentages(pattern)
numLabels = len(classpercentages)
num_stream_cycles = cycles_in_stream * numLabels
num_ref_patterns = patterns_in_ref * numLabels

seed_ref = 123 + sum([ord(char) for char in pattern])
seed_stream = 456 + sum([ord(char) for char in pattern])
if not os.path.isdir(os.path.join(core_path, folder_name)):
    print('creating directory : ' + os.path.join(core_path, folder_name))
    os.makedirs(os.path.join(core_path, folder_name))

if not os.path.isdir(os.path.join(core_path, folder_name, sub_folder)):
    print('creating directory : ' + os.path.join(core_path, folder_name, sub_folder))
    os.makedirs(os.path.join(core_path, folder_name, sub_folder))

# ! --------------------------------------------------------------------------------------------- COMPUTE DTW

# ------------------------------- reference patterns load data
REF = np.load(os.path.join(core_path, folder_name, fileREF))
labelsREF = REF[:, 0]
REF = REF[:, 1:]

for s in range(len(sets_list)):
    print(f'SET :: {sets_list[s]}')
    if sets_list[s] in ['validation', 'train', 'test']:
        refIDs = refMedoids(os.path.join(core_path, folder_name, fileREF))
    else:
        refIDs = np.arange(REF.shape[0], dtype=int)
    for j in range(num_streams_set[s]):
        # ------------------------------- load streams
        if sets_list[s] == 'test' and pattern == 'gunpoint':
            cycles_in_stream_test = 20
            fileSTREAM = f'STREAM_cycles-per-label-{cycles_in_stream_test}'

        file = f'{fileSTREAM}_set-{sets_list[s]}_id-{j}.npy'
        STREAM = np.load(os.path.join(core_path, folder_name, file))
        labelsSTREAM = STREAM[:, 1]
        STREAM = STREAM[:, 0]
        print(np.shape(STREAM))

        for r in refIDs:
            if pattern in ['cbf', 'two_patterns2', 'two_patterns', 'rational', 'synthetic_control']:
                fileRP = os.path.join(sub_folder,
                                      f'rpMat-{sets_list[s]}_length-{length}_noise-{noise_level}_warp-{warp_level}'
                                      f'_shift-{shift_level}_outliers-0_ref-id-{r}_stream-id-{j}.npy')
            else:
                fileRP = os.path.join(sub_folder,
                                      f'rpMat-{sets_list[s]}_ref-id-{r}_stream-id-{j}.npy')

            if os.path.isfile(os.path.join(core_path, folder_name, fileRP)) is False:
                print(f'Computing Recurrence Plot between (ref, stream_id) = ({r}, {j})')
                ref_arr = REF[r, :].reshape((-1, 1))
                stream_arr = STREAM.reshape((-1, 1))
                distMat = cdist(ref_arr, stream_arr)
                np.save(os.path.join(core_path, folder_name, fileRP), distMat)
            else:
                print(f'Computing Recurrence Plot between (ref, stream_id) = ({r}, {j})')
