import os

import tensorflow as tf
import numpy as np
import random

import yaml

os.environ['PYTHONHASHSEED'] = str(42)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

from utils.functions import Paths
from utils.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.CNN.model import get_model, optimizer

with open('conf.yaml') as file:
  conf_data = yaml.safe_load(file)


dataset = conf_data['dataset']
window_size = conf_data['window_size']
dataset_type = conf_data['dataset_type']
rho = conf_data['rho']

base_pattern = True if len(conf_data['pattern_name']) != 0 else False
pattern_name = conf_data['pattern_name']

dataset_name = dataset if not base_pattern else dataset+'_base'
y_dim = specs[dataset_name]['y_dim']
x_dim = specs[dataset_name]['x_dim']
channels = specs[dataset_name]['channels']

if pattern_name == 'FULL':
    channels = 5
elif len(pattern_name) > 0:
    channels = len(pattern_name)


parameters = conf_data['parameters']
column_scale = True if parameters['scaler_dim'] != [0, 1] else False


paths = Paths(dataset, dataset_type, rho, window_size, base_pattern, pattern_name, column_scale=column_scale)

data_path = paths.get_data_path()
weight_dir = paths.get_weight_dir()

model_name = f'{dataset_type}_CNN_{window_size}'
if column_scale:
    model_name += '_column_scale'

NN = Network(data_path,
             x_dim=(x_dim, window_size, channels), y_dim=y_dim,
             model_name=f'{model_name}.hdf5')

NN.init_model(get_model, parameters, optimizer, create_generator)
# NN.train(epochs=100, save_path=weight_dir, from_checkpoint=False)
NN.evaluate(weights_dir=weight_dir)
# NN.check_pattern(weights_dir=weight_dir, dataset_name=dataset)
NN.explain(weights_dir=weight_dir, dataset_name=dataset)
NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

