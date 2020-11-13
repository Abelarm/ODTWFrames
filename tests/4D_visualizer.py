from glob import glob
from os.path import join
from random import choice

import numpy as np
import matplotlib.pyplot as plt

from dataset.files import RefPattern, TimeSeries
from utils.functions import get_id_interval, Paths
from utils.specification import specs, multi_rho, cmap

dataset = 'rational'
dataset_type = 'DTW'
base_pattern = False
pattern_name = ''
dataset_name = dataset if not base_pattern else dataset+'_base'
always_custom = True

rho = 'multi'
window_size = 5


rho_name = 'rho '
rho_name += rho if not base_pattern else rho+'_base'
rho_name += f'_{pattern_name}' if len(pattern_name) > 0 else ''
if dataset_type == 'RP':
    rho = ''

channels = specs[dataset_name]['channels']
if len(pattern_name) > 0:
    channels = len(pattern_name)
length = specs[dataset_name]['x_dim']

paths = Paths(dataset, dataset_type, rho, window_size, base_pattern, pattern_name, core_path='../')

data_path = paths.get_data_path()
beginning_path = paths.get_beginning_path()

if base_pattern:
    interval = [1129, 1134]
    ts_id = 0
    class_id = 2
else:
    interval = []
    ts_id = 0

len_generator = len(join(data_path, '*'))

if len(interval) == 0 or ts_id is None:
    idx = choice(range(len_generator))
    file_nn = glob(join(data_path, 'test', '*'))[idx]
    ts_id, interval = get_id_interval(file_nn.split('/')[-1])
else:
    filename_to_search = f'X:{ts_id}_{interval[0]}-{interval[0]+window_size}|Y:{class_id}.npy'
    file_nn = glob(join(data_path, 'test', filename_to_search))[0]

x = np.load(file_nn)

if dataset == 'gunpoint':
    ref_name = 'REF_num-5.npy'
    stream_name = f'STREAM_cycles-per-label-20_set-test_id-{ts_id}.npy'
else:
    stream_name = f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_'\
                  f'set-test_id-{ts_id}.npy'
    ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'

if base_pattern:
    ref_name = f'BASE_REF_len-{length}_noise-5_num-1.npy'
    if len(pattern_name) > 0:
        ref_name = f'BASE_REF_len-{length}_noise-5_num-1_{pattern_name}.npy'

ref = RefPattern(f'{beginning_path}/{ref_name}')
t = TimeSeries(f'{beginning_path}/{stream_name}')
timeseries = t.timeseries

fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(2, 1, bottom=0.6)

f_axi_0 = fig.add_subplot(gs[0, :])
f_axi_0.plot(timeseries)
f_axi_0.axvline(x=interval[0], linewidth=2, color='r')
f_axi_0.axvline(x=interval[1], linewidth=2, color='r')
f_axi_0.axis(xmin=interval[0]-75, xmax=interval[1]+75)

ref_ids = specs[dataset_name]['ref_id']
if len(pattern_name) > 0:
    ref_ids = range(len(pattern_name))

print(x.shape)
classes = [i for i in range(specs[dataset_name]['y_dim'])]
rhos = multi_rho
gs2 = fig.add_gridspec(5, 5*len(classes), top=0.68,
                        hspace=0.05)
for i in range(x.shape[2]):

    for j in range(x.shape[3]):

        dtw = x[:, :, i, j]
        if j != 0:
            y = y + 5
        else:
            y = 0
        f_axi_1 = fig.add_subplot(gs2[i, y:y + 5])
        img = f_axi_1.imshow(dtw, cmap=cmap, origin='lower', aspect='auto')
        if i == 0:
            f_axi_1.set_title(f'Class: {classes[j%len(classes)]}')
        if y == 0:
            f_axi_1.set_ylabel(f'rho: {rhos[i%len(rhos)]}')

        f_axi_1.set_xticks([])
        f_axi_1.set_yticks([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.12, 0.01, 0.64])
fig.colorbar(img, cmap='plasma', cax=cbar_ax)
plt.show()