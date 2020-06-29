import os

import tensorflow as tf
import numpy as np
import random

os.environ['PYTHONHASHSEED'] = str(42)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

from utils.functions import Paths
from utils.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.CNN.model import get_model, optimizer

dataset = 'gunpoint'
dataset_type = 'DTW'
rho = 'multi'
window_size = 1
base_pattern = False
pattern_name = ''


dataset_name = dataset if not base_pattern else dataset+'_base'
y_dim = specs[dataset_name]['y_dim']
x_dim = specs[dataset_name]['x_dim']
channels = specs[dataset_name]['channels']

if pattern_name == 'FULL':
    channels = 5
elif len(pattern_name) > 0:
    channels = len(pattern_name)


parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False


paths = Paths(dataset, dataset_type, rho, window_size, base_pattern, pattern_name)

data_path = paths.get_data_path()
weight_dir = paths.get_weight_dir()


window_size = channels if rho == 'multi' else window_size
channels = 3 if rho == 'multi' else channels

ws_name = 3 if rho == 'multi' else window_size

NN = Network(data_path,
             x_dim=(x_dim, window_size, channels), y_dim=y_dim,
             model_name=f'{dataset_type}_CNN_{ws_name}.hdf5')

NN.init_model(get_model, parameters, optimizer, create_generator)
# NN.train(epochs=100, save_path=weight_dir, from_checkpoint=False)
# NN.evaluate(weights_dir=weight_dir)
# NN.check_pattern(weights_dir=weight_dir, dataset_name=dataset)
# NN.explain(weights_dir=weight_dir, dataset_name=dataset)
NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
# NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

