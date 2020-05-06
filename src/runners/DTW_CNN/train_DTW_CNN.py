from comet_ml import Experiment

from utils.functions import Paths
from utils.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.CNN.model import get_model, optimizer

dataset = 'cbf'
dataset_type = 'DTW'
rho = '0.100'
window_size = 5
base_pattern = True
pattern_name = 'AB'


dataset_name = dataset if not base_pattern else dataset+'_base'
y_dim = specs[dataset_name]['y_dim']
x_dim = specs[dataset_name]['x_dim']
channels = specs[dataset_name]['channels']
if len(pattern_name) > 0:
    channels = len(pattern_name)

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False


paths = Paths(dataset, dataset_type, rho, window_size, base_pattern, pattern_name)

data_path = paths.get_data_path()
weight_dir = paths.get_weight_dir()

# Add the following code anywhere in your machine learning file
project_name = f'DTW_CNN_{dataset_name}'
if len(pattern_name) > 0:
    project_name = f'DTW_CNN_{dataset}_base_compare'

experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
                        project_name=project_name, workspace="luigig")

NN = Network(data_path,
             x_dim=(x_dim, window_size, channels), y_dim=y_dim,
             model_name=f'DTW_CNN_{window_size}.hdf5', experiment=experiment)

NN.init_model(get_model, parameters, optimizer, create_generator)
NN.train(epochs=100, save_path=weight_dir, from_checkpoint=False)
NN.evaluate(weights_dir=weight_dir)
# NN.check_pattern(weights_dir=weight_dir, dataset_name=dataset)
NN.explain(weights_dir=weight_dir, dataset_name=dataset)
NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

experiment.end()
