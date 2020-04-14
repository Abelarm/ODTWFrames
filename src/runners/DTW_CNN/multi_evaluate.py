from itertools import product

from comet_ml import Experiment

from dataset.specification import specs
from models.CNN.model import get_model, optimizer
from models.generator_proxy import create_generator
from models.network import Network

dataset = ['rational']
rho = ['0.500']
window_size = [5, 15]

parameters = dict()
parameters['batch_size'] = 32
parameters['preprocessing'] = True
parameters['reload_images'] = False

# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
#                         project_name=f'DTW_CNN_{dataset}', workspace="luigig")

for ds, r, ws in product(dataset, rho, window_size):

    print(f'Creating dataset for: {ds} - {r} - {ws}')


    core_path = '../../..'
    beggining_path = f'{core_path}/data/{ds}/'
    dataset_type = 'DTW'

    y_dim = specs[ds]['y_dim']
    x_dim = specs[ds]['x_dim']

    NN = Network(f'{beggining_path}/rho {r}/{dataset_type}_{ws}',
                 x_dim=(x_dim, ws, y_dim), y_dim=y_dim,
                 model_name=f'DTW_CNN_{ws}.hdf5', experiment=None)

    NN.init_model(get_model, parameters, optimizer, create_generator)
    # NN.train(epochs=1)
    # NN.evaluate(weights_dir=f'{core_path}/Network_weights/{dataset}/rho {rho}')
    NN.summary_experiments(weights_dir=f'{core_path}/Network_weights/{ds}/rho {r}', dataset_name=ds)
    NN.error_analysis(weights_dir=f'{core_path}/Network_weights/{ds}/rho {r}', dataset_name=ds)

    # experiment.end()