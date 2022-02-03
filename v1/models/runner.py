import os
from comet_ml import Experiment
import tensorflow as tf
import numpy as np
import random

from models.generator_proxy import create_generator
from models.network import Network

os.environ['PYTHONHASHSEED'] = str(42)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


def train(dataset, project_name, paths, x_dim, y_dim, get_model, parameters, optimizer,
          epochs=100, from_checkpoint=False, lr=0, track_experiment=False):

    data_path = paths.get_data_path()
    weight_dir = paths.get_weight_dir()

    if track_experiment:
        experiment = Experiment(api_key="",
                                project_name=project_name, workspace="",
                                disabled=True)
    else:
        experiment = None

    NN = Network(data_path, x_dim=x_dim, y_dim=y_dim, experiment=experiment)

    NN.init_model(get_model, parameters, optimizer, create_generator)
    NN.train(epochs=epochs, from_checkpoint=from_checkpoint, lr=lr, save_path=weight_dir)
    NN.evaluate(weights_dir=weight_dir)
    NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
    # NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)

    experiment.end()


def evaluate(dataset, project_name, paths, x_dim, y_dim, get_model, parameters, optimizer,
             evaluating=True, summary=True, error=True, explain=True):

    data_path = paths.get_data_path()
    weight_dir = paths.get_weight_dir()

    experiment = Experiment(api_key="tIjRDRXwqoq2RgkME4epGXp1C",
                            project_name=project_name, workspace="luigig",
                            disabled=True)

    NN = Network(data_path, x_dim=x_dim, y_dim=y_dim, experiment=experiment)

    NN.init_model(get_model, parameters, optimizer, create_generator)
    if evaluating:
        NN.evaluate(weights_dir=weight_dir)
    if summary:
        NN.summary_experiments(weights_dir=weight_dir, dataset_name=dataset)
    if error:
        NN.error_analysis(weights_dir=weight_dir, dataset_name=dataset)
    if explain:
        NN.explain(weights_dir=weight_dir, dataset_name=dataset)

    experiment.end()