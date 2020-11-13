import yaml

from runners.runner import train
from utils.functions import Paths
from utils.specification import specs
from models.ResNet1D.model import get_model, optimizer

with open('conf.yaml') as file:
  conf_data = yaml.safe_load(file)


dataset = conf_data['dataset']
window_size = conf_data['window_size']
dataset_type = conf_data['dataset_type']
rho = conf_data['rho']
post_processing = conf_data['post_processing']

base_pattern = True if len(conf_data['pattern_name']) != 0 else False
pattern_name = conf_data['pattern_name']

network_type = 'ResNet'
appendix_name = None


paths = Paths(dataset, dataset_type, rho, window_size, base_pattern=base_pattern, pattern_name=pattern_name,
              network_type=network_type, appendix_name=appendix_name, post_processing=post_processing)


dataset_name = dataset if not base_pattern else dataset+'_base'
y_dim = specs[dataset_name]['y_dim']

y_dim = y_dim if rho else 1

x_dim = specs[dataset_name]['x_dim']
channels = specs[dataset_name]['channels']

if pattern_name == 'FULL':
    channels = 5
elif len(pattern_name) > 0:
    channels = len(pattern_name)


parameters = conf_data['parameters']
column_scale = True if parameters['scaler_dim'] != [0, 1] else False

project_name = f'{dataset_type}_{network_type}_{dataset}'

train(dataset=dataset,
      project_name=project_name,
      paths=paths,
      x_dim=(window_size, y_dim),
      y_dim=y_dim,
      get_model=get_model,
      parameters=parameters,
      optimizer=optimizer)
