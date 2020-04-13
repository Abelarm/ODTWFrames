from comet_ml import Experiment

from dataset.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.CRNN.model import get_model, optimizer

dataset = 'cbf'
core_path = '../../..'
beggining_path = f'{core_path}/data/{dataset}/'
dataset_type = 'DTW'
rho = '0.100'
y_dim = specs[dataset]['y_dim']
x_dim = specs[dataset]['x_dim']
window_size = 25

parameters = dict()
parameters['batch_size'] = 8
parameters['preprocessing'] = True

# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
#                         project_name="DTW_CNN", workspace="luigig")


NN = Network(f'{beggining_path}/rho {rho}/{dataset_type}_{window_size}',
             x_dim=(x_dim, window_size, y_dim), y_dim=y_dim,
             model_name=f'DTW_CRNN_{window_size}.hdf5', experiment=None)
NN.init_model(get_model, parameters, optimizer, create_generator)
NN.train(epochs=1)
NN.evaluate(weights_dir=f'{core_path}/Network_weights/{dataset}')



# experiment.end()