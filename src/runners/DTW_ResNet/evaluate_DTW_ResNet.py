import yaml

from runners.runner import evaluate
from utils.functions import Paths
from utils.specification import specs
from models.ResNet.model_100k import get_model, optimizer

with open('conf.yaml') as file:
    conf_data = yaml.safe_load(file)

dataset = conf_data['dataset']
window_size = conf_data['window_size']
dataset_type = conf_data['dataset_type']
rho = conf_data['rho']

base_pattern = True if len(conf_data['pattern_name']) != 0 else False
pattern_name = conf_data['pattern_name']

network_type = 'CNN'
appendix_name = 'ResNet_100k'

paths = Paths(dataset, dataset_type, rho, window_size, base_pattern=base_pattern, pattern_name=pattern_name,
              network_type=network_type, appendix_name=appendix_name)

dataset_name = dataset if not base_pattern else dataset + '_base'
y_dim = specs[dataset_name]['y_dim']
x_dim = specs[dataset_name]['x_dim']
channels = specs[dataset_name]['channels']

if pattern_name == 'FULL':
    channels = 5
elif len(pattern_name) > 0:
    channels = len(pattern_name)

parameters = conf_data['parameters']
column_scale = True if parameters['scaler_dim'] != [0, 1] else False

project_name = f'{dataset_type}_{network_type}_{dataset}'

evaluate(dataset=dataset,
         project_name=project_name,
         paths=paths,
         x_dim=(x_dim, window_size, channels),
         y_dim=y_dim,
         get_model=get_model,
         parameters=parameters,
         optimizer=optimizer,
         evaluating=True,
         summary=True,
         error=False,
         explain=False)
