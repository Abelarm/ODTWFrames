import os

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from dataset.generate_files.DataGenerator.utils import compute_classpercentages
from dataset.generate_files.utils.distance import refMedoids, compute_odtw_distance_matrix
from utils.specification import specs

db_spec = {
    'noise_level': 5,  # std white noise (rate)
    'warp_level': 10,
    'shift_level': 10,
    'cycles_in_stream': 10,  # number of patterns per label in the stream
    'patterns_in_ref': 10  # number of  pattern per label in the reference set
}


def compute_distance(mat_type, pattern, sub_pattern, sub_pattern_names, rho):

    # *                                                     data folders and files
    core_path = '../data'
    folder_name = f'{pattern}'
    sub_folder = mat_type if mat_type == 'RP' else f'rho {rho}'

    length = specs[pattern]['x_dim']
    mat_type_name = mat_type.lower()

    if sub_pattern:
        sub_folder += '_base'
    if len(sub_pattern_names) > 0:
        sub_folder += f'_{sub_pattern_names}'

    # stream type
    sets_list = ['train', 'validation', 'test']  # streaming sets

    if pattern == 'gunpoint':
        num = 5
        length = 150
        fileREF = f'REF_num-{num}.npy'
        fileSTREAM = f'STREAM_cycles-per-label-{db_spec["cycles_in_stream"]}'
    else:

        fileSTREAM = f'STREAM_length-{length}_noise-{db_spec["noise_level"]}_warp-{db_spec["warp_level"]}' \
                     f'_shift-{db_spec["shift_level"]}'\
                     f'_outliers-0_cycles-per-label-{db_spec["cycles_in_stream"]}'

        fileREF = f'REF_length-{length}_noise-{db_spec["noise_level"]}_warp-{db_spec["warp_level"]}' \
                  f'_shift-{db_spec["shift_level"]}'\
                  f'_outliers-0_num-{db_spec["patterns_in_ref"]}.npy'

    if sub_pattern:
        if len(sub_pattern_names) > 0:
            fileREF = f'BASE_REF_len-{length}_noise-5_num-1_{sub_pattern_names}.npy'
        else:
            fileREF = f'BASE_REF_len-{length}_noise-5_num-1.npy'

    num_streams_set = specs[pattern]['max_stream_id']  # number of streams in each set

    # ! --------------------------------------------------------------------------------------------- INITIALIZATION
    classpercentages = compute_classpercentages(pattern)
    numLabels = len(classpercentages)
    num_stream_cycles = db_spec["cycles_in_stream"] * numLabels
    num_ref_patterns = db_spec["patterns_in_ref"] * numLabels

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
            data = np.load(os.path.join(core_path, folder_name, fileREF))
            refIDs = refMedoids(data)
        else:
            refIDs = np.arange(REF.shape[0], dtype=int)
        for j in tqdm(range(num_streams_set[s])):
            # ------------------------------- load streams
            if sets_list[s] == 'test' and pattern == 'gunpoint':
                cycles_in_stream_test = 20
                fileSTREAM = f'STREAM_cycles-per-label-{cycles_in_stream_test}'

            file = f'{fileSTREAM}_set-{sets_list[s]}_id-{j}.npy'
            STREAM = np.load(os.path.join(core_path, folder_name, file))
            labelsSTREAM = STREAM[:, 1]
            STREAM = STREAM[:, 0]
            # print(np.shape(STREAM))

            rho_string = '' if mat_type == 'RP' else f'rho-{rho}_'
            for r in refIDs:

                if pattern in ['cbf', 'two_patterns2', 'two_patterns', 'rational', 'synthetic_control']:

                    fileRP = os.path.join(sub_folder,
                                          f'{mat_type_name}Mat-{sets_list[s]}_length-{length}'
                                          f'_noise-{db_spec["noise_level"]}'
                                          f'_warp-{db_spec["warp_level"]}'
                                          f'_shift-{db_spec["shift_level"]}'
                                          f'_outliers-0_'
                                          f'{rho_string}'
                                          f'ref-id-{r}_stream-id-{j}.npy')
                else:
                    fileRP = os.path.join(sub_folder,
                                          f'{mat_type_name}Mat-{sets_list[s]}_'
                                          f'{rho_string}'
                                          f'ref-id-{r}_stream-id-{j}.npy')

                if os.path.isfile(os.path.join(core_path, folder_name, fileRP)) is False:
                    # print(f'Computing Recurrence Plot between (ref, stream_id) = ({r}, {j})')
                    if mat_type == 'DTW':
                        distMat = compute_odtw_distance_matrix(REF[r, :], STREAM, float(rho) ** (1.0 / length))
                    elif mat_type == 'RP':
                        ref_arr = REF[r, :].reshape((-1, 1))
                        stream_arr = STREAM.reshape((-1, 1))
                        distMat = cdist(ref_arr, stream_arr)
                    np.save(os.path.join(core_path, folder_name, fileRP), distMat)
                # else:
                #    print(f'Recurrence Plot between (ref, stream_id) = ({r}, {j}) already computed')
