import os

from comet_ml import Experiment
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

dataset = 'cbf'
dataset_type = 'RP'
rho = '0.100'
window_size = 5
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

# Add the following code anywhere in your machine learning file
project_name = f'{dataset_type}_CNN_{dataset_name}'
if len(pattern_name) > 0:
    project_name = f'{dataset_type}_CNN_{dataset}'

experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
                        project_name=project_name, workspace="luigig", disabled=False)

NN = Network(data_path,
             x_dim=(x_dim, window_size, channels), y_dim=y_dim,
             model_name=f'{dataset_type}_CNN_{window_size}.hdf5', experiment=experiment)

NN.init_model(get_model, parameters, optimizer, create_generator)
NN.train(epochs=100, save_path=weight_dir, from_checkpoint=False)
NN.evaluate(weights_dir=weight_dir)
# NN.check_pattern(weights_dir=weight_dir, dataset_name=dataset)
NN.explain(weights_dir=weight_dir, dataset_name=dataset)
NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

experiment.end()
