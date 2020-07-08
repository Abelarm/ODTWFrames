from os import path

from dataset.basePattern.create import BasePattern
from dataset.dataset import Dataset
from dataset.files import TimeSeries
from dataset.generate_files.compute_distance import compute_distance
from dataset.generate_files.generate_database import generate_database
from utils.specification import multi_rho


def generate(dataset,
             mat_type,
             beginning_path,
             rho,
             window_size,
             n_classes,
             length,
             max_stream_id,
             base_pattern=False,
             pattern_name='',
             path_class=None):

    if dataset == 'gunpoint':
        ref_name = 'REF_num-5.npy'
        stream_name = 'STREAM_cycles-per-label-10_set'
    else:
        ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'
        stream_name = f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set'
        if not path.isfile(f'{beginning_path}/{ref_name}') and \
                not path.isfile(f'{beginning_path}/{stream_name}'):
            print("pattern REF and pattern STREAM \n "
                  "MUST ALREADY EXSITS must be one of the followings. \n "
                  " \t 1) arma              ----> 8 classes \n "
                  " \t 2) synthetic_control ----> 6 classes \n "
                  " \t 3) sines             ----> 5 classes \n "
                  " \t 4) kohlerlorenz      ----> 5 classes \n "
                  " \t 5) cbf               ----> 3 classes \n "
                  " \t 6) two_patterns      ----> 4 classes \n "
                  " \t 7) rational          ----> 4 classes \n "
                  " \t 8) seasonal          ----> 4 classes")
            exit(1)
        print("========= CREATING RAW DATA")
        generate_database(dataset)

    if base_pattern:
        ref_name = f'BASE_REF_len-{length}_noise-5_num-1.npy'
        if len(pattern_name) > 0:
            ref_name = f'BASE_REF_len-{length}_noise-5_num-1_{pattern_name}.npy'

    if base_pattern:
        t = TimeSeries(
            f'../data/{dataset}/{stream_name}-train_id-0.npy')
        timeseries = t.timeseries
        base = BasePattern(length, timeseries.min(), timeseries.max())
        base.compute_pattern(pattern_name)
        base.save(f'../data/{dataset}')

    print("========= CALCULATING DISTANCE MATRIX")
    if rho == 'multi':
        for rho_val in multi_rho:
            compute_distance(mat_type, dataset, base_pattern, pattern_name, rho_val)
            # pass

    save_path_image = path_class.get_data_path()
    dtw_starting_path = path_class.get_dtw_path()

    ds = Dataset(mat_type,
                 f'{beginning_path}/{ref_name}',
                 f'{beginning_path}/{stream_name}-train_id-*.npy',
                 'train',
                 dtw_starting_path,
                 rho,
                 window_size=window_size,
                 classes=[x + 1 for x in range(n_classes)],
                 max_id=max_stream_id[0] - 1)

    ref_ids = ds.create_image_dataset(save_path_image, base_pattern=base_pattern)
    # ds.create_series_image_dataset(f'{beginning_path}/rho {rho}/CRNN_DTW_{window_size}', 3)
    if not path.exists(f'{beginning_path}/TS_{window_size}/train'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beginning_path}/TS_{window_size}/train')

    ds = Dataset(mat_type,
                 f'{beginning_path}/{ref_name}',
                 f'{beginning_path}/{stream_name}-validation_id-*.npy',
                 'validation',
                 dtw_starting_path,
                 rho,
                 window_size=window_size,
                 classes=[x + 1 for x in range(n_classes)],
                 max_id=max_stream_id[1] - 1)

    ds.create_image_dataset(save_path_image, base_pattern=base_pattern)
    # ds.create_series_image_dataset(f'{beginning_path}/rho {rho}/CRNN_DTW_{window_size}', 3)
    if not path.exists(f'{beginning_path}/TS_{window_size}/validation'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beginning_path}/TS_{window_size}/validation')

    if 'gunpoint' in beginning_path:
        stream_name = 'STREAM_cycles-per-label-20_set'
    ds = Dataset(mat_type,
                 f'{beginning_path}/{ref_name}',
                 f'{beginning_path}/{stream_name}-test_id-*.npy',
                 'test',
                 dtw_starting_path,
                 rho,
                 window_size=window_size,
                 classes=[x + 1 for x in range(n_classes)],
                 max_id=max_stream_id[2] - 1)

    print(f'Using selected ids: {ref_ids}')
    ds.create_image_dataset(save_path_image, base_pattern=base_pattern, ref_ids=ref_ids)
    # ds.create_series_image_dataset(f'{beginning_path}/rho {rho}/CRNN_DTW_{window_size}', 3, ref_ids)
    if not path.exists(f'{beginning_path}/TS_{window_size}/test'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beginning_path}/TS_{window_size}/test')
