from comet_ml import Experiment

from dataset.specification import specs
from models.CNN.model import get_model, optimizer
from models.generator_proxy import create_generator
from models.network import Network

dataset = 'gunpoint'
core_path = '../../..'
beggining_path = f'{core_path}/data/{dataset}/'
dataset_type = 'DTW'
rho = '0.001'
y_dim = specs[dataset]['y_dim']
x_dim = specs[dataset]['x_dim']
window_size = 5

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False

# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
#                         project_name=f'DTW_CNN_{dataset}', workspace="luigig")


NN = Network(f'{beggining_path}/rho {rho}/{dataset_type}_{window_size}',
             x_dim=(x_dim, window_size, y_dim), y_dim=y_dim,
             model_name=f'DTW_CNN_{window_size}.hdf5', experiment=None)

NN.init_model(get_model, parameters, optimizer, create_generator)
# NN.train(epochs=1)
# NN.evaluate(weights_dir=f'{core_path}/Network_weights/{dataset}/rho {rho}')
# NN.summary_experiments(weights_dir=f'{core_path}/Network_weights/{dataset}/rho {rho}', dataset_name=dataset)
NN.error_analysis(weights_dir=f'{core_path}/Network_weights/{dataset}/rho {rho}', dataset_name=dataset)

# experiment.end()