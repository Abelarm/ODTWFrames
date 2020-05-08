from comet_ml import Experiment

from utils.functions import Paths
from utils.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.ResNet1D.model import get_model, optimizer

dataset = 'cbf'
dataset_type = 'TS'
window_size = 5

y_dim = specs[dataset]['y_dim']
x_dim = specs[dataset]['x_dim']

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False


paths = Paths(dataset, dataset_type, False, window_size, False, '')

data_path = paths.get_data_path()
weight_dir = paths.get_weight_dir()

print(f'=========== Saving the weights into: {weight_dir} ===========')

# Add the following code anywhere in your machine learning file
project_name = f'TS_ResNet_{dataset}'

experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
                        project_name=project_name, workspace="luigig",
                        disabled=False)

NN = Network(data_path,
             x_dim=(window_size, 1), y_dim=y_dim,
             model_name=f'TS_ResNet_{window_size}.hdf5', experiment=experiment)

NN.init_model(get_model, parameters, optimizer, create_generator)
NN.train(epochs=100, save_path=weight_dir, from_checkpoint=False)
NN.evaluate(weights_dir=weight_dir)
NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

experiment.end()
