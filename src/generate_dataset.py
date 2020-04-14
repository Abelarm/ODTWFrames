from itertools import product

from dataset.generator import generate
from dataset.specification import specs

dataset = ['cbf']
core_path = '../../..'
beggining_path = f'{core_path}/data/{dataset}/'
rho = ['0.100']
window_size = [5, 15, 25]
base = True

for ds, r,  w_s in product(dataset, rho, window_size):

    path = f'../data/{ds}/'

    print(f'Creating dataset for: {ds} - {path} - {r} - {w_s}')

    classes = specs[ds]['y_dim']

    generate(path, r, w_s, classes, base_pattern=base)