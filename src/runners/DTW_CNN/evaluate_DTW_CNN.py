from utils.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.CNN.model import get_model, optimizer

dataset = 'rational'
core_path = '../../..'
beginning_path = f'{core_path}/data/{dataset}/'
dataset_type = 'DTW'
rho = '0.100'
window_size = 5
base_pattern = True

dataset_name = dataset if not base_pattern else dataset+'_base'
y_dim = specs[dataset_name]['y_dim']
x_dim = specs[dataset_name]['x_dim']
channels = specs[dataset_name]['channels']

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False

data_path = f'{beginning_path}/rho {rho}/{dataset_type}_{window_size}'
weight_dir = f'{core_path}/Network_weights/{dataset}/rho {rho}'
if base_pattern:
    data_path = f'{beginning_path}/rho {rho}_base/{dataset_type}_{window_size}'
    weight_dir = f'{core_path}/Network_weights/{dataset}/rho {rho}_base/'

# Add the following code anywhere in your machine learning file
project_name = f'DTW_CNN_{dataset_name}'
# experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
#                         project_name=project_name, workspace="luigig")

NN = Network(data_path,
             x_dim=(x_dim, window_size, channels), y_dim=y_dim,
             model_name=f'DTW_CNN_{window_size}.hdf5', experiment=None)

NN.init_model(get_model, parameters, optimizer, create_generator)
# NN.train(epochs=10, save_path=weight_dir, from_checkpoint=False)
# NN.evaluate(weights_dir=weight_dir)
# NN.check_pattern(weights_dir=weight_dir, dataset_name=dataset)
NN.explain(weights_dir=weight_dir, dataset_name=dataset)
# NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
# NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

# experiment.end()