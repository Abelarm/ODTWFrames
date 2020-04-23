from itertools import product

from dataset.generator import generate
from utils.specification import specs

dataset = ['cbf']
core_path = '../../..'
beginning_path = f'{core_path}/data/{dataset}/'
rho = ['0.100']
window_size = [5]
base = True

for ds, r,  w_s in product(dataset, rho, window_size):

    path = f'../data/{ds}/'

    print(f'Creating dataset for: {ds} - {path} - {r} - {w_s}')

    classes = specs[ds]['y_dim']
    ds_name = ds if not base else ds+'_base'
    max_stream_id = specs[ds_name]['max_stream_id']

    generate(path, r, w_s, classes, max_stream_id=max_stream_id, base_pattern=base)