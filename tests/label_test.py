import operator
import re
from collections import Counter
from glob import glob
from os.path import join

from PIL import Image
import numpy as np

from dataset.files import TimeSeries
from utils.functions import Paths
from utils.specification import specs

dataset_type = 'DTW'
dataset = 'gunpoint'
base_pattern = False
dataset_name = dataset if not base_pattern else dataset+'_base'

rho = '0.100'
window_size = 5
rho_name = 'rho '
pattern_name = ''
rho_name += rho if not base_pattern else rho+'_base'
always_custom = True


if dataset == 'gunpoint':
    ref_name = 'REF_num-5.npy'
    stream_name = 'STREAM_cycles-per-label-20_set-test_id-*.npy'
elif base_pattern:
    ref_name = 'BASE_REF_len-100_noise-5_num-1.npy'
    stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-*.npy'
else:
    ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'
    stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-*.npy'


paths = Paths(dataset, dataset_type, rho, window_size, base_pattern, pattern_name, network_type='CNN', core_path='../')
beginning_path = paths.get_beginning_path()

for files in glob(join(beginning_path, stream_name)):
    t = TimeSeries(files)
    ts_id = t.get_properties()['id']
    if int(ts_id) > 8:
        print(f"skipping the id: {ts_id}")
        continue
    print(f'Checking STREAM ID: {ts_id}')

    num_channels = specs[dataset_name]['channels']
    length = specs[dataset_name]['x_dim']
    if 2 < num_channels < 5 and not always_custom:
        final_string = f'*/{ts_id}_*'
        interval_exp = rf'{ts_id}_(.+)\.'
        class_exp = rf'test\/(.+)\/{ts_id}'
    else:
        final_string = f'X:{ts_id}_*'
        interval_exp = rf'X:{ts_id}_(.*)\|'
        class_exp = r'Y:(.+)\.'

    training_files = glob(join(paths.get_data_path(), 'test', final_string))
    assert len(training_files) > 0
    print(f"Found: {len(training_files)} files")
    for t_f in training_files:
        interval = re.search(interval_exp, t_f, re.IGNORECASE).group(1)
        interval = list(map(int, interval.split('-')))

        selected_labels = t.labels[interval[0]:interval[1]]

        tmp_labels = dict(Counter(selected_labels))
        real_class = int(max(tmp_labels.items(), key=operator.itemgetter(1))[0])

        calc_class = re.search(class_exp, t_f, re.IGNORECASE).group(1)
        calc_class = int(calc_class)

        assert real_class == calc_class

        if '.png' in t_f:
            loaded_x = Image.open(t_f)
            loaded_x = np.array(loaded_x.getdata())
            loaded_x = loaded_x.reshape(specs[dataset_name]['x_dim'], window_size, -1)
        else:
            loaded_x = np.load(t_f)
        print(f'Checking the dtwMat file of STREAM file {t_f}')
        chan = 0
        for c in specs[dataset_name]['ref_id']:

            if dataset_type == 'RP':
                dtw_path = f'rpMat-test_length-{length}_noise-5_warp-10_shift-10_outliers-0' \
                           f'_ref-id-{c}_stream-id-{ts_id}.npy'
            else:
                if dataset_name == 'gunpoint':
                    dtw_path = f'dtwMat-test_rho-0.100_ref-id-{c}_stream-id-{ts_id}.npy'
                else:

                    dtw_path = f'dtwMat-test_length-{length}_noise-5_warp-10_shift-10_outliers-0' \
                               f'_rho-0.100_ref-id-{c}_stream-id-{ts_id}.npy'

            loaded_dtw = np.load(join(paths.get_dtw_path(), dtw_path))
            sliced_dtw = loaded_dtw[:, interval[0]:interval[1]]

            assert (loaded_x[:, :, chan] == sliced_dtw).all()
            chan += 1
