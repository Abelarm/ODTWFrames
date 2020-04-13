from itertools import product

from dataset.generator import generate

dataset_name = ['rational']
n_classes = {'rational': 4}
beggining_path = f'../data/{dataset_name}/'
rho = ['0.001', '0.100', '0.500']
window_size = [5, 15, 25]

for n, r,  w_s in product(dataset_name, rho, window_size):

    path = f'../data/{n}/'

    print(f'Creating dataset for: {n} - {path} - {r} - {w_s}')

    classes = n_classes[n]

    generate(path, r, w_s, classes)