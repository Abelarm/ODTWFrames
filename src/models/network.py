import re
from os import listdir, makedirs
from os.path import join, exists

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report

from models.dataGenerator import DataGenerator
from networkAnalysis.errors import analysis
from networkAnalysis.summary import plot_acc_loss, plot_roc_auc, plot_confusion_matrix, plot_class_probabilities

core_path = '../../..'

class Network:
    root_dir = None
    x_dim = None
    y_dim = None
    experiment = None
    model = None
    parameters = None
    train_generator = None
    validate_generator = None
    test_generator = None
    y_pred = None

    def __init__(self, root_dir, x_dim, y_dim, model_name, experiment=None):

        self.root_dir = root_dir
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.experiment = experiment
        self.model_name = model_name

        if re.search('rho (.*)/', self.root_dir, re.IGNORECASE):
            self.rho = re.search('rho (.*)/', self.root_dir, re.IGNORECASE).group(1)
        else:
            self.rho = None

        if re.search('_base', self.root_dir, re.IGNORECASE):
            self.base_pattern = True
            # since there is base in the rho path
            self.rho = self.rho.replace('_base', '')
        else:
            self.base_pattern = False

    def init_model(self, model_function, parameters, optimizer, generator_function):

        self.parameters = parameters

        self.train_generator, \
        self.validate_generator, \
        self.test_generator, \
        self.test_generator_analysis = generator_function(self.root_dir, self.x_dim, self.y_dim,
                                                          **self.parameters)

        self.model = model_function(self.x_dim, self.y_dim)

        self.optimizer = optimizer
        print(optimizer.get_config())

        self.model.compile(optimizer=self.optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def train(self, epochs, save_path='.', from_checkpoint=False, lr=0):

        self.epochs = epochs

        total_train = 0
        total_val = 0
        total_test = 0
        if exists(f'{self.root_dir}/train/1'):
            for i in range(1, self.y_dim + 1):
                total_train += len(listdir(f'{self.root_dir}/train/{i}'))
                total_val += len(listdir(f'{self.root_dir}/validation/{i}'))
                total_test += len(listdir(f'{self.root_dir}/test/{i}'))
        else:
            total_train += len(listdir(f'{self.root_dir}/train/'))
            total_val += len(listdir(f'{self.root_dir}/validation/'))
            total_test += len(listdir(f'{self.root_dir}/test/'))

        print("Total training images:", total_train)
        print("Total validation images:", total_val)
        print("Total test images:", total_test)

        if not exists(save_path):
            print(f'Creating directories to {save_path}')
            makedirs(save_path)

        filepath = join(save_path, self.model_name)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=5,
                                       patience=5,
                                       min_lr=0.5e-7)

        callbacks_list = [checkpoint, lr_reducer]

        if from_checkpoint:
            print('Restarting the training from checkpoint')
            self.model.load_weights(filepath)
            self.optimizer.__init__(learning_rate=float(lr))
            self.model.compile(optimizer=self.optimizer,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

        if self.experiment:
            with self.experiment.train():
                self.history = self.model.fit(
                    self.train_generator,
                    steps_per_epoch=self.train_generator.__len__(),
                    epochs=epochs,
                    validation_data=self.validate_generator,
                    validation_steps=self.validate_generator.__len__(),
                    callbacks=callbacks_list
                )

            self.experiment.log_parameters(self.parameters)
        else:
            self.history = self.model.fit(
                self.train_generator,
                steps_per_epoch=self.train_generator.__len__(),
                epochs=epochs,
                validation_data=self.validate_generator,
                validation_steps=self.validate_generator.__len__(),
                callbacks=callbacks_list
            )

    def evaluate(self, weights_dir):

        self.model.load_weights(join(weights_dir, self.model_name))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        loss, acc = self.model.evaluate(self.validate_generator)
        print(f'VALIDATION Evaluation loss: {loss} - accuracy: {acc}')

        loss, acc = self.model.evaluate(self.test_generator)
        print(f'TEST Evaluation loss: {loss} - accuracy: {acc}')

        metrics = {
            'loss': loss,
            'accuracy': acc
        }

        if self.experiment:
            self.experiment.log_metrics(metrics)

        if self.y_pred is None:
            self.y_pred = self.model.predict(self.test_generator_analysis, verbose=1)

        if type(self.test_generator) is DataGenerator:
            y_true = self.test_generator.classes - 1
        else:
            y_true = self.test_generator.classes

        print(y_true.shape)
        print(self.y_pred.shape)

        print(y_true[0])
        print(self.y_pred[0])

        y_pred_arg = np.argmax(self.y_pred, axis=1)

        print('Classification Report')
        self.target_names = [str(i) for i in range(self.y_dim)]
        print(classification_report(y_true, y_pred_arg, target_names=self.target_names))

        if self.experiment:
            self.experiment.log_other('Classification Report',
                                      classification_report(y_true, y_pred_arg,
                                                            target_names=self.target_names))

        enc = OneHotEncoder(sparse=False)
        y_true = enc.fit_transform(y_true.reshape(-1, 1))

        if self.experiment:
            self.experiment.log_confusion_matrix(y_true, self.y_pred)

    def summary_experiments(self, weights_dir, dataset_name):

        self.model.load_weights(join(weights_dir, self.model_name))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        model_name_tmp = self.model_name.replace('.hdf5', '').replace('_CNN', '').replace('_1DCNN', '')
        if self.rho:
            save_dir = f'{core_path}/experiment_summaries/{dataset_name}/rho {self.rho}/{model_name_tmp}/'
            if self.base_pattern:
                save_dir = f'{core_path}/experiment_summaries/{dataset_name}/rho {self.rho}_base/{model_name_tmp}/'
        else:
            save_dir = f'{core_path}/experiment_summaries/{dataset_name}/{model_name_tmp}/'
        makedirs(save_dir, exist_ok=True)

        self.target_names = [str(i) for i in range(self.y_dim)]

        if hasattr(self, 'history'):
            plot_acc_loss(self.history, self.epochs, save_dir)

        if type(self.test_generator) is DataGenerator:
            y_true = self.test_generator.classes - 1
        else:
            y_true = self.test_generator.classes

        if self.y_pred is None:
            self.y_pred = self.model.predict(self.test_generator_analysis, verbose=1)

        with open(join(save_dir, 'metrics.txt'), 'w') as f:
            y_pred_arg = np.argmax(self.y_pred, axis=1)
            f.write(classification_report(y_true, y_pred_arg, digits=4, target_names=self.target_names))
            print(f'Saving {join(save_dir, "metrics.txt")}')
            f.close()

        plot_roc_auc(self.y_dim, y_true, self.y_pred, save_dir)

        plot_confusion_matrix(y_true, self.y_pred, self.target_names, save_dir)

        if len(self.x_dim) > 2:
            # images
            plot_class_probabilities(self.test_generator_analysis, self.y_dim, dataset_name,
                                     self.rho, model_name_tmp, self.x_dim[1], self.y_pred, save_dir)
        else:
            # timeseries
            plot_class_probabilities(self.test_generator_analysis, self.y_dim, dataset_name,
                                     self.rho, model_name_tmp, self.x_dim[0], self.y_pred, save_dir)

    def error_analysis(self, weights_dir, dataset_name):

        self.model.load_weights(join(weights_dir, self.model_name))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        if self.y_pred is None:
            self.y_pred = self.model.predict(self.test_generator_analysis, verbose=1)

        if type(self.test_generator) is DataGenerator:
            y_true = self.test_generator.classes - 1
        else:
            y_true = self.test_generator.classes

        print(y_true.shape)
        print(self.y_pred.shape)

        print(y_true[0])
        print(self.y_pred[0])

        model_name_tmp = self.model_name.replace('.hdf5', '').replace('_CNN', '').replace('_1DCNN', '')
        if self.rho:
            save_dir = f'{core_path}/error_analysis/{dataset_name}/rho {self.rho}/{model_name_tmp}/'
            if self.base_pattern:
                save_dir = f'{core_path}/error_analysis/{dataset_name}/rho {self.rho}_base/{model_name_tmp}/'
        else:
            save_dir = f'{core_path}/error_analysis/{dataset_name}/{model_name_tmp}/'

        analysis(self.test_generator_analysis, y_true, self.y_pred, save_dir=save_dir,
                 dataset_name=dataset_name)
