import re
from random import choice

import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator

from dataset.files import RefPattern, TimeSeries
from models.dataGenerator import DataGenerator
from utils.specification import specs

core_path = '../data'

dataset = 'cbf'
base_pattern = False
dataset_name = dataset if not base_pattern else dataset+'_base'

rho = '0.100'
window_size = 5
rho_name = 'rho '
rho_name += rho if not base_pattern else rho+'_base'

channels = specs[dataset_name]['channels']
length = specs[dataset_name]['x_dim']

if 2 < channels < 5:
    datagen = ImageDataGenerator(rescale=1./255,
                                 featurewise_center=False,
                                 featurewise_std_normalization=False)

    train_generator = datagen.flow_from_directory(
        directory=f'{core_path}/{dataset_name}/{rho_name}/DTW_{window_size}/test',
        target_size=(100, window_size),
        color_mode="rgb" if channels != 4 else "rgba",
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
        seed=42
    )
    class_idx = 0
    stream_id_exp = rf'[0-9]+\/(.+)_'
    interval_exp = rf'[0-9]+_(.*)\.'

else:
    x_dim = (specs[dataset_name]['x_dim'], window_size, channels)
    train_generator = DataGenerator(f'{core_path}/{dataset}/{rho_name}/DTW_{window_size}/test',
                                    dim=x_dim,
                                    n_classes=specs[dataset_name]['y_dim'],
                                    to_fit=True,
                                    shuffle=False,
                                    batch_size=1,
                                    scaler=None,
                                    preprocessing=False)

    class_idx = -5
    stream_id_exp = rf'X:(.+)_'
    interval_exp = rf'X:[0-9]+_(.*)\|'


len_generator = len(train_generator)
idx = choice(range(len_generator))
print(f'selected sample with ID: {idx} \n')
x, y = train_generator[idx]

print(f"X shape: {x[0].shape}")
print(f"Y shape: {y[0].shape}")
print(f"All classes: {np.unique(train_generator.classes)} \n")

filename = train_generator.filenames[idx]
print(f'Calculated from: {filename}')

print(f"Real value of Y: {filename[class_idx]}, one_hot version: {y}")

stream_id = re.search(stream_id_exp, filename, re.IGNORECASE).group(1)

if dataset == 'gunpoint':
    ref_name = 'REF_num-5.npy'
    stream_name = f'STREAM_cycles-per-label-20_set-test_id-{stream_id}.npy'
else:
    stream_name = f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-{stream_id}.npy'
    ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'

if base_pattern:
    ref_name = f'BASE_REF_len-{length}_noise-5_num-1.npy'

ref = RefPattern(f'../data/{dataset}/{ref_name}')
t = TimeSeries(
    f'../data/{dataset}/{stream_name}')
timeseries = t.timeseries

interval = re.search(interval_exp, filename, re.IGNORECASE).group(1)
interval = list(map(int, interval.split('-')))

fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(3, x[0].shape[-1])

f_axi_0 = fig.add_subplot(gs[0, :])
f_axi_0.plot(timeseries)
f_axi_0.axvline(x=interval[0], linewidth=2, color='r')
f_axi_0.axvline(x=interval[1], linewidth=2, color='r')
f_axi_0.axis(xmin=interval[0]-75, xmax=interval[1]+75)

ref_ids = specs[dataset_name]['ref_id']
for i, c in enumerate(ref_ids):

    f_axi_1 = fig.add_subplot(gs[1, i])
    f_axi_1.plot(ref.lab_patterns[c]['pattern'])
    f_axi_1.set_xticks([])
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