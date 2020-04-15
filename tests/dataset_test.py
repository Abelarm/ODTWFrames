from random import choice

import numpy as np
from keras_preprocessing.image import ImageDataGenerator

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


len_generator = len(train_generator)
idx = choice(range(len_generator))
print(f'selected ID: {idx} \n')
x, y = train_generator[idx]

print(f"X shape: {x[0].shape}")
print(f"Y shape: {y[0].shape}")
print(f"All classes: {np.unique(train_generator.classes)} \n")

filename = train_generator.filenames[idx]
print(f'Calculated from: {filename}')

print(f"Real value of Y: {filename[class_idx]}, one_hot version: {y}")