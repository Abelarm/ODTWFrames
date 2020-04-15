import operator
import re
from collections import Counter
from glob import glob
from os.path import join

from dataset.files import TimeSeries
from utils.specification import specs

core_path = '../data'

dataset = 'cbf'
base_pattern = True
dataset_name = dataset if not base_pattern else dataset+'_base'

if dataset == 'gunpoint':
    ref_name = 'REF_num-5.npy'
    stream_name = 'STREAM_cycles-per-label-20_set-test_id-*.npy'
elif base_pattern:
    ref_name = 'BASE_REF_len-100_noise-5_num-1.npy'
    stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-*.npy'
else:
    ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'
    stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-*.npy'

rho = '0.100'
window_size = 5
rho_name = 'rho '
rho_name += rho if not base_pattern else rho+'_base'

for files in glob(join(core_path, dataset, stream_name)):
    t = TimeSeries(files)
    id = t.get_properties()['id']
    print(f'Checking ID: {id}')

    num_channels = specs[dataset_name]['channels']
    if 2 < num_channels < 5:
        final_string = f'*/{id}_*'
        interval_exp = rf'{id}_(.+)\.'
        class_exp = rf'test\/(.+)\/{id}'
    else:
        final_string = f'X:{id}_*'
        interval_exp = rf'X:{id}_(.*)\|'
        class_exp = r'Y:(.+)\.'

    training_files = glob(join(core_path, dataset, rho_name, f'DTW_{window_size}', 'test', final_string))
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





