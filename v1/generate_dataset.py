from itertools import product

from dataset.generator import generate
from utils.functions import Paths
from utils.specification import specs

dataset_array = ['cbf']
rho_array = ['0.100']
window_size_array = [5]
base_pattern = False
pattern_name = 'ABC' if base_pattern else ''
dataset_type = 'DTW'
post_processing = ''

for dataset, rho, window_size in product(dataset_array, rho_array, window_size_array):

    classes = specs[dataset]['y_dim']
    length = specs[dataset]['x_dim']
    ds_name = dataset if not base_pattern else dataset+'_base'
    max_stream_id = specs[ds_name]['max_stream_id']

    paths = Paths(dataset, dataset_type, rho, window_size, base_pattern, pattern_name, network_type='', core_path='..')
    beginning_path = paths.get_beginning_path()

    print(f'Creating {dataset_type} dataset for: {dataset} - {beginning_path} - {rho} - {window_size} - {pattern_name}')

    generate(dataset,
             dataset_type, beginning_path, rho, window_size, classes,
             length=length,
             max_stream_id=max_stream_id,
             base_pattern=base_pattern,
             pattern_name=pattern_name,
             path_class=paths,
             post_processing=post_processing)
