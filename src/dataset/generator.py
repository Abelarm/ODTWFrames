from os import path

from dataset.dataset import Dataset


def generate(beginning_path, rho, window_size, n_classes, max_stream_id,
             base_pattern=False,
             pattern_name='',
             path_class=None):

    length = 100

    if 'gunpoint' in beginning_path:
        ref_name = 'REF_num-5.npy'
        stream_name = 'STREAM_cycles-per-label-20_set'
        length = 150
    else:
        stream_name = f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set'
        ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'

    if base_pattern:
        ref_name = f'BASE_REF_len-{length}_noise-5_num-1.npy'
        if len(pattern_name) > 0:
            ref_name = f'BASE_REF_len-{length}_noise-5_num-1_{pattern_name}.npy'

    save_path_image = path_class.get_data_path()
    dtw_starting_path = path_class.get_dtw_path()

    ds = Dataset(f'{beginning_path}/{ref_name}',
                 f'{beginning_path}/{stream_name}-train_id-*.npy',
                 'train',
                 dtw_starting_path,
                 rho,
                 window_size=window_size,
                 classes=[x+1 for x in range(n_classes)],
                 max_id=max_stream_id[0]-1)

    ref_ids = ds.create_image_dataset(save_path_image, base_pattern=base_pattern)
    # ds.create_series_image_dataset(f'{beginning_path}/rho {rho}/CRNN_DTW_{window_size}', 3)
    if not path.exists(f'{beginning_path}/TS_{window_size}/train'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beginning_path}/TS_{window_size}/train')

    ds = Dataset(f'{beginning_path}/{ref_name}',
                 f'{beginning_path}/{stream_name}-validation_id-*.npy',
                 'validation',
                 dtw_starting_path,
                 rho,
                 window_size=window_size,
                 classes=[x+1 for x in range(n_classes)],
                 max_id=max_stream_id[1]-1)

    ds.create_image_dataset(save_path_image, base_pattern=base_pattern)
    # ds.create_series_image_dataset(f'{beginning_path}/rho {rho}/CRNN_DTW_{window_size}', 3)
    if not path.exists(f'{beginning_path}/TS_{window_size}/validation'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beginning_path}/TS_{window_size}/validation')

    if 'gunpoint' in beginning_path:
        stream_name = 'STREAM_cycles-per-label-20_set'
    ds = Dataset(f'{beginning_path}/{ref_name}',
                 f'{beginning_path}/{stream_name}-test_id-*.npy',
                 'test',
                 dtw_starting_path,
                 rho,
                 window_size=window_size,
                 classes=[x+1 for x in range(n_classes)],
                 max_id=max_stream_id[2]-1)

    print(f'Using selected ids: {ref_ids}')
    ds.create_image_dataset(save_path_image, base_pattern=base_pattern, ref_ids=ref_ids)
    # ds.create_series_image_dataset(f'{beginning_path}/rho {rho}/CRNN_DTW_{window_size}', 3, ref_ids)
    if not path.exists(f'{beginning_path}/TS_{window_size}/test'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beginning_path}/TS_{window_size}/test')
