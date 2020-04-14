from comet_ml import Experiment

from dataset.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.CNN.model import get_model, optimizer

dataset = 'cbf'
core_path = '../../..'
beggining_path = f'{core_path}/data/{dataset}/'
dataset_type = 'DTW'
rho = '0.100'
y_dim = specs[dataset]['y_dim']
x_dim = specs[dataset]['x_dim']
window_size = 5
base_pattern = True

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False

data_path = f'{beggining_path}/rho {rho}/{dataset_type}_{window_size}'
weight_dir = f'{core_path}/Network_weights/{dataset}/rho {rho}'
channel_dim = y_dim
if base_pattern:
    data_path = f'{beggining_path}/rho {rho}_base/{dataset_type}_{window_size}'
    weight_dir = f'{core_path}/Network_weights/{dataset}/rho {rho}_base/'
    channel_dim = 7

# Add the following code anywhere in your machine learning file
project_name = f'DTW_CNN_{dataset}'
if base_pattern:
    project_name = f'DTW_CNN_{dataset}_base'

experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
                        project_name=project_name, workspace="luigig")

NN = Network(data_path,
             x_dim=(x_dim, window_size, channel_dim), y_dim=y_dim,
             model_name=f'DTW_CNN_{window_size}.hdf5', experiment=experiment)

NN.init_model(get_model, parameters, optimizer, create_generator)
NN.train(epochs=10, save_path=weight_dir, from_checkpoint=False)
NN.evaluate(weights_dir=weight_dir)
NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

# experiment.end()