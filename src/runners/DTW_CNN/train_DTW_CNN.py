from comet_ml import Experiment

from dataset.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.CNN.model import get_model, optimizer

dataset = 'gunpoint'
beggining_path = f'../../../data/{dataset}/'
dataset_type = 'DTW'
rho = '0.500'
y_dim = specs[dataset]['y_dim']
x_dim = specs[dataset]['x_dim']
window_size = 15

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
                        project_name=f'DTW_CNN_{dataset}', workspace="luigig")


NN = Network(f'{beggining_path}/rho {rho}/{dataset_type}_{window_size}',
             x_dim=(x_dim, window_size, y_dim), y_dim=y_dim,
             model_name=f'DTW_CNN_{window_size}.hdf5', experiment=experiment)

NN.init_model(get_model, parameters, optimizer, create_generator)
NN.train(epochs=9, save_path=f'../Network_weights/{dataset}/rho {rho}', from_checkpoint=True, lr='5e-07')
NN.evaluate(weights_dir=f'../Network_weights/{dataset}/rho {rho}')
NN.summary_experiments(weights_dir=f'../Network_weights/{dataset}/rho {rho}/', dataset_name=dataset)
NN.error_analysis(weights_dir=f'../Network_weights/{dataset}/rho {rho}/', dataset_name=dataset)

# experiment.end()