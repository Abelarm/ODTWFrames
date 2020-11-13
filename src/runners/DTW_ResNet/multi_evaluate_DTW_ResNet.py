from itertools import product

from runners.trainer import evaluate
from utils.functions import Paths
from utils.specification import specs
from models.CNN.model import get_model, optimizer


dataset_arr = ['rational']
dataset_type = 'DTW'
rho_arr = ['0.100']
window_size_arr = [5]
base_pattern = True
pattern_name_arr = ['A', 'B', 'AB']

for dataset, rho, window_size, pattern_name in product(dataset_arr, rho_arr, window_size_arr, pattern_name_arr):

    print(f'============= EVALUATING FOR: {dataset}|ws:{window_size}|rho:{rho}|pattern_name:{pattern_name}')

    dataset_name = dataset if not base_pattern else dataset+'_base'
    y_dim = specs[dataset_name]['y_dim']
    x_dim = specs[dataset_name]['x_dim']
    channels = specs[dataset_name]['channels']

    if pattern_name == 'FULL':
        channels = 5
    elif len(pattern_name) > 0:
        channels = len(pattern_name)

    parameters = dict()
    parameters['batch_size'] = 32
    parameters['preprocessing'] = True
    parameters['reload_images'] = False

    network_type = 'ResNet'
    appendix_name = '1M'

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
             summary=True,
             error=False,
             explain=False)
