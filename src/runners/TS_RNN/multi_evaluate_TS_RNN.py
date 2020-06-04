from itertools import product
from utils.functions import Paths, Singleton
from utils.specification import specs
from models.generator_proxy import create_generator
from models.network import Network
from models.RNN.model import get_model, optimizer

dataset_arr = ['gunpoint']
dataset_type = 'TS'
window_size_arr = [5, 15, 25]

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False


for dataset, window_size in product(dataset_arr, window_size_arr):

    print(f'============= EVALUATING FOR: {dataset}|ws:{window_size}')

    y_dim = specs[dataset]['y_dim']
    x_dim = specs[dataset]['x_dim']

    paths = Paths(dataset, dataset_type, False, window_size, False, '')

    data_path = paths.get_data_path()
    weight_dir = paths.get_weight_dir()

    NN = Network(data_path,
                 x_dim=(window_size, 1), y_dim=y_dim,
                 model_name=f'TS_RNN_{window_size}.hdf5', experiment=None)

    NN.init_model(get_model, parameters, optimizer, create_generator)
    # NN.train(epochs=1, save_path=weight_dir, from_checkpoint=False)
    # NN.evaluate(weights_dir=weight_dir)
    NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
    # NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

    del NN
    del paths



