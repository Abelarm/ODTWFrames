import yaml

from models.get_models import get_model_function
from models.runner import train
from utils.functions import Paths
from utils.specification import specs

with open('conf.yaml') as file:
    conf_data = yaml.safe_load(file)

dataset = conf_data['dataset']
window_size = conf_data['window_size']
dataset_type = conf_data['dataset_type']
rho = conf_data['rho']

base_pattern = True if len(conf_data['pattern_name']) != 0 else False
pattern_name = conf_data['pattern_name']

network_type = conf_data['network_type']
appendix_name = conf_data['appendix_name']

paths = Paths(dataset, dataset_type, rho, window_size, base_pattern=base_pattern, pattern_name=pattern_name,
              network_type=network_type, appendix_name=appendix_name, core_path='../')

dataset_name = dataset if not base_pattern else dataset + '_base'
y_dim = specs[dataset_name]['y_dim']
x_dim = specs[dataset_name]['x_dim']
channels = specs[dataset_name]['channels']

if pattern_name == 'FULL':
    channels = 5
elif len(pattern_name) > 0:
    channels = len(pattern_name)

parameters = conf_data['parameters']

batch_size = parameters['batch_size']
epochs = conf_data['epochs']

column_scale = True if parameters['scaler_dim'] != [0, 1] else False

if dataset_type == 'DTW':
    project_name = f'{dataset_type}_CNN_{dataset}'
elif dataset_type == 'TS':
    project_name = f'{dataset_type}_{network_type}_{dataset}'

get_model, optimizer = get_model_function(dataset_type, network_type, appendix_name)

train(dataset=dataset,
      project_name=project_name,
      paths=paths,
      x_dim=(x_dim, window_size, channels),
      y_dim=y_dim,
      get_model=get_model,
      parameters=parameters,
      optimizer=optimizer,
      epochs=epochs)
