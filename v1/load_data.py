## we load multi-variate datasets into the environment
from sktime.utils.data_io import load_from_arff_to_dataframe
import pandas as pd
import numpy as np

import matplotlib

from dataset.generate_files.utils.distance import compute_odtw_distance_matrix

matplotlib.use('tkagg')
from matplotlib import pyplot as plt

classes = ['Badminton', 'Running', 'Standing', 'Walking']

X_train, y_train = load_from_arff_to_dataframe("../data/basic_motions/BasicMotions_TRAIN.arff")
X_train.head()
X_test,y_test = load_from_arff_to_dataframe("../data/basic_motions/BasicMotions_TEST.arff")


# def calculate_metoids(X, Y, class_name):
#
#     ids = np.squeeze(np.argwhere(Y==class_name))
#
#     for i in ids:
#         print(X[i])

def plot_and_save(X, type):
    full_array = None
    rng = np.random.default_rng()
    random_indexes = list(range(X.shape[0]))
    rng.shuffle(random_indexes)
    for i in random_indexes:
        tmp_array = pd.concat([X['dim_0'][i].to_frame(),X['dim_1'][i].to_frame(),
                               X['dim_2'][i].to_frame(),X['dim_3'][i].to_frame(),
                               X['dim_4'][i].to_frame(),X['dim_5'][i].to_frame(),
                               pd.Series([y_train[i]]*100)],axis=1,ignore_index=True).to_numpy()

        if full_array is not None:
            full_array = np.vstack((full_array, tmp_array))
        else:
            full_array = tmp_array

    np.save(f'../data/basic_motions/STREAM_set-{type}_id-0', full_array)
    plt.plot(full_array[:, :-1])
    plt.show()

    dtw = compute_odtw_distance_matrix(full_array[:100, :-1].astype('float'), full_array[:, :-1].astype('float'), 0.1)

    np.save(f'../data/basic_motions/rho 0.100/dtwMat-{type}_stream-id-0', dtw)
    plt.imshow(dtw, aspect='auto', origin='lower')
    plt.show()



#calculate_metoids(X_train, y_train, 'Standing')

plot_and_save(X_train, 'train')
plot_and_save(X_test, 'test')