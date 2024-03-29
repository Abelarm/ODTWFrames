from utils.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.OneDCNN.model import get_model, optimizer

dataset = 'rational'
core_path = '../../..'
beginning_path = f'{core_path}/data/{dataset}/'
dataset_type = 'TS'
y_dim = specs[dataset]['y_dim']
window_size = 25

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True


# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
#                         project_name=f'DTW_CNN_{dataset}', workspace="luigig")


NN = Network(f'{beginning_path}/{dataset_type}_{window_size}',
             x_dim=(window_size, 1), y_dim=y_dim,
             model_name=f'TS_1DCNN_{window_size}.hdf5', experiment=None)

NN.init_model(get_model, parameters, optimizer, create_generator)
# NN.train(epochs=1)
# NN.evaluate(weights_dir=f'../Network_weights/{dataset}')
# NN.summary_experiments(weights_dir=f'{core_path}/Network_weights/{dataset}/{dataset_type}', dataset_name=dataset)
NN.error_analysis(weights_dir=f'{core_path}/Network_weights/{dataset}/{dataset_type}', dataset_name=dataset)


# experiment.end()