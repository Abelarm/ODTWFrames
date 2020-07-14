from random import choice

import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator

from dataset.files import RefPattern, TimeSeries
from models.dataGenerator import DataGenerator
from utils.functions import get_id_interval, Paths
from utils.specification import specs


dataset = 'rational'
dataset_type = 'DTW'
base_pattern = True
pattern_name = 'ABC'
dataset_name = dataset if not base_pattern else dataset+'_base'
always_custom = True

rho = 'multi'
window_size = 1


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


if 2 < channels < 5 and not always_custom:
    datagen = ImageDataGenerator(rescale=1./255,
                                 featurewise_center=False,
                                 featurewise_std_normalization=False)

    train_generator = datagen.flow_from_directory(
        directory=f'{data_path}/test',
        target_size=(100, window_size),
        color_mode="rgb" if channels != 4 else "rgba",
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )
    class_idx = 0

else:
    second_dim = 5 if rho == 'multi' else window_size
    x_dim = (specs[dataset_name]['x_dim'], second_dim, channels)
    train_generator = DataGenerator(f'{data_path}/test',
                                    dim=x_dim,
                                    n_classes=specs[dataset_name]['y_dim'],
                                    to_fit=True,
                                    shuffle=False,
                                    batch_size=1,
                                    scaler=None,
                                    preprocessing=False)

    class_idx = -5

if base_pattern:
    interval = [1129, 1134]
    ts_id = 0
    class_id = 2
else:
    interval = []
    ts_id = 0
len_generator = len(train_generator)

if len(interval) == 0 or ts_id is None:
    idx = choice(range(len_generator))
else:
    filename_to_search = f'X:{ts_id}_{interval[0]}-{interval[0]+window_size}|Y:{class_id}.npy'
    idx = train_generator.filenames.index(filename_to_search)

print(f'selected sample with ID: {idx} \n')
x, y = train_generator[idx]

print(f"X shape: {x[0].shape}")
print(f"Y shape: {y[0].shape}")
print(f"All classes: {np.unique(train_generator.classes)} \n")

filename = train_generator.filenames[idx]
print(f'Calculated from: {filename}')

print(f"Real value of Y: {filename[class_idx]}, one_hot version: {y}")

stream_id, interval = get_id_interval(filename)

if dataset == 'gunpoint':
    ref_name = 'REF_num-5.npy'
    stream_name = f'STREAM_cycles-per-label-20_set-test_id-{stream_id}.npy'
else:
    stream_name = f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_'\
                  f'set-test_id-{stream_id}.npy'
    ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'

if base_pattern:
    ref_name = f'BASE_REF_len-{length}_noise-5_num-1.npy'
    if len(pattern_name) > 0:
        ref_name = f'BASE_REF_len-{length}_noise-5_num-1_{pattern_name}.npy'

ref = RefPattern(f'{beginning_path}/{ref_name}')
t = TimeSeries(f'{beginning_path}/{stream_name}')
timeseries = t.timeseries

fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(3, x[0].shape[-1])

f_axi_0 = fig.add_subplot(gs[0, :])
f_axi_0.plot(timeseries)
f_axi_0.axvline(x=interval[0], linewidth=2, color='r')
f_axi_0.axvline(x=interval[1], linewidth=2, color='r')
f_axi_0.axis(xmin=interval[0]-75, xmax=interval[1]+75)

ref_ids = specs[dataset_name]['ref_id']
if len(pattern_name) > 0:
    ref_ids = range(len(pattern_name))

for i, c in enumerate(ref_ids):

    f_axi_1 = fig.add_subplot(gs[1, i])
    f_axi_1.plot(ref.lab_patterns[c]['pattern'])
    f_axi_1.set_xticks([])
    if i != 0:
        f_axi_1.set_yticks([])

    dtw = x[0][:, :, i]
    f_axi_2 = fig.add_subplot(gs[2, i])
    img = f_axi_2.imshow(dtw, cmap='plasma', origin='lower', aspect='auto')
    f_axi_2.set_xticks([])
    f_axi_2.set_yticks([])

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.12, 0.01, 0.64])
fig.colorbar(img, cmap='plasma', cax=cbar_ax)
plt.show()
