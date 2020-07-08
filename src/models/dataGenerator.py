from glob import glob

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.utils import Sequence
from tqdm import tqdm


class DataGenerator(Sequence):
    """Generates data for Keras networks'
    Sequence based data generator. Suitable for building data generator.

    Can be used for generate data from 3D matrix (DTW-images) and 2D matrix (time series)
    """
    def __init__(self,
                 sequence_path,
                 dim,
                 n_classes,
                 categorical=True,
                 to_fit=True,
                 batch_size=32,
                 shuffle=True,
                 preprocessing=False,
                 scaler_dim=(0, 1),
                 scaler=None):
        """

        :param sequence_path: Path where is located the sequence
        :param dim: (tuple) of X dimensions
        :param n_classes: number of classes of the data
        :param categorical: if transform to categorical
        :param to_fit: it is used for fit purpose of for prediction
        :param batch_size: batch size of the single sample
        :param shuffle: True o False
        :param preprocessing: if pre-process the data
        :param scaler: scaler used in other DataGenerator, fit on train data, transform on test,val data
        """

        self.sequence_path = sequence_path
        self.categorical = categorical
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.normalize = preprocessing
        self.scaler_dim = scaler_dim
        self.all_samples_name = []

        for x in glob(f'{self.sequence_path}/X*'):
            self.all_samples_name.append(x)

        self.all_samples_name = np.asarray(self.all_samples_name)
        self.on_epoch_end()

        if self.categorical:
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder = self.encoder.fit(self.classes)

        if self.normalize:
            if not scaler:
                self.scaler = StandardScaler()
                print('Fitting the scaler')
                for n in tqdm(self.all_samples_name):
                    x = np.load(f'{n}')
                    if len(x.shape) > 2:
                        res_x = x.reshape((x.shape[self.scaler_dim[0]]*x.shape[self.scaler_dim[1]], -1))
                        self.scaler.partial_fit(res_x)
                    else:
                        self.scaler.partial_fit(x)
                print('Fitted complete')
            else:
                self.scaler = scaler

    @property
    def filenames(self):
        return list(map(lambda x: x.split("/")[-1], self.all_samples_name))

    def __len__(self):
        """Denotes the number of batches per epoch

        :return: number of batches per epoch
        """
        if (len(self.all_samples_name) / self.batch_size) % 1 == 0:
            return int(len(self.all_samples_name) / self.batch_size)
        else:
            return int(np.floor(len(self.all_samples_name) / self.batch_size)) + 1

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        if (index + 1) * self.batch_size > len(self.indexes):
            indexes = self.indexes[index * self.batch_size:]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.all_samples_name[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    @property
    def classes(self):

        y = np.empty((len(self.all_samples_name), 1), dtype=int)

        for i, id in enumerate(self.all_samples_name):
            y[i] = int(id.split('|Y:')[1].replace('.npy', ''))

        return y

    def on_epoch_end(self):
        """Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.all_samples_name))
        if self.shuffle:
            np.random.seed(42)
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images

        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((len(list_IDs_temp), *self.dim))

        for i, path in enumerate(list_IDs_temp):
            tmp_x = np.load(f'{path}')
            if self.normalize:
                if len(tmp_x.shape) > 2:
                    orignal_shape = tmp_x.shape
                    res_x = tmp_x.reshape((tmp_x.shape[self.scaler_dim[0]] * tmp_x.shape[self.scaler_dim[1]], -1))
                    res_x = self.scaler.transform(res_x)
                    X[i] = res_x.reshape(orignal_shape)
                else:
                    X[i] = self.scaler.transform(tmp_x)
            else:
                X[i] = tmp_x

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks

        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((len(list_IDs_temp), self.n_classes), dtype=int)

        for i, id in enumerate(list_IDs_temp):
            tmp_y = id.split('|Y:')[1].replace('.npy', '')
            if self.categorical:
                y[i] = self.encoder.transform(np.asarray(int(tmp_y)).reshape(-1, 1))
            else:
                y[i] = tmp_y

        return y
