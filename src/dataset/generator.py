from os import path

from dataset.dataset import Dataset


def generate(beggining_path, rho, window_size, n_classes, base_pattern=False):

    if 'gunpoint' in beggining_path:
        ref_name = 'REF_num-5.npy'
        stream_name = 'STREAM_cycles-per-label-10_set'
    elif base_pattern:
        ref_name = 'BASE_REF_len-100_noise-5_num-1.npy'
        stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set'
    else:
        ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'
        stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set'

    save_path_image = f'{beggining_path}/rho {rho}/DTW_{window_size}'
    if base_pattern:
        save_path_image = f'{beggining_path}/rho {rho}/DTW_{window_size}_base'

    ds = Dataset(f'{beggining_path}{ref_name}',
                 f'{beggining_path}{stream_name}-train_id-*.npy',
                 'train',
                 f'{beggining_path}/rho {rho}',
                 rho,
                 window_size=window_size,
                 classes=[x+1 for x in range(n_classes)])

    ref_ids = ds.create_image_dataset(save_path_image)
    # ds.create_series_image_dataset(f'{beggining_path}/rho {rho}/CRNN_DTW_{window_size}', 3)
    if not path.exists(f'{beggining_path}/TS_{window_size}/train'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beggining_path}/TS_{window_size}/train')

    ds = Dataset(f'{beggining_path}{ref_name}',
                 f'{beggining_path}{stream_name}-validation_id-*.npy',
                 'validation',
                 f'{beggining_path}/rho {rho}',
                 rho,
                 window_size=window_size,
                 classes=[x+1 for x in range(n_classes)])

    ds.create_image_dataset(save_path_image)
    # ds.create_series_image_dataset(f'{beggining_path}/rho {rho}/CRNN_DTW_{window_size}', 3)
    if not path.exists(f'{beggining_path}/TS_{window_size}/validation'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beggining_path}/TS_{window_size}/validation')

    if 'gunpoint' in beggining_path:
        stream_name = 'STREAM_cycles-per-label-20_set'
    ds = Dataset(f'{beggining_path}{ref_name}',
                 f'{beggining_path}{stream_name}-test_id-*.npy',
                 'test',
                 f'{beggining_path}/rho {rho}',
                 rho,
                 window_size=window_size,
                 classes=[x+1 for x in range(n_classes)])

    print(f'Using selected ids: {ref_ids}')
    ds.create_image_dataset(save_path_image, ref_ids=ref_ids)
    # ds.create_series_image_dataset(f'{beggining_path}/rho {rho}/CRNN_DTW_{window_size}', 3, ref_ids)
    if not path.exists(f'{beggining_path}/TS_{window_size}/test'):
        print('====== CREATING SERIES DATASET ====== ')
        ds.create_series_dataset(f'{beggining_path}/TS_{window_size}/test')