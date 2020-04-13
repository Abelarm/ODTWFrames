from comet_ml import Experiment

from dataset.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.RNN.model import get_model, optimizer

dataset = 'gunpoint'
core_path = '../../..'
beggining_path = f'{core_path}/data/{dataset}/'
dataset_type = 'TS'
y_dim = specs[dataset]['y_dim']
window_size = 5

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True

# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
#                         project_name="DTW_CNN", workspace="luigig")


NN = Network(f'{beggining_path}/{dataset_type}_{window_size}',
             x_dim=(window_size, 1), y_dim=y_dim,
             model_name=f'TS_1DCNN_{window_size}.hdf5', experiment=None)

NN.init_model(get_model, parameters, optimizer, create_generator)
NN.train(epochs=60, save_path=f'{core_path}/Network_weights/{dataset}/{dataset_type}', from_checkpoint=True, lr='1e-05')
NN.evaluate(weights_dir=f'{core_path}/Network_weights/{dataset}/{dataset_type}')
NN.summary_experiments(weights_dir=f'{core_path}/Network_weights/{dataset}/{dataset_type}', dataset_name=dataset)
NN.error_analysis(weights_dir=f'{core_path}/Network_weights/{dataset}/{dataset_type}', dataset_name=dataset)
