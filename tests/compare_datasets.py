from glob import glob
from os.path import join

from PIL import Image
import numpy as np

core_path = '../data'
dataset_name = 'cbf'
rho = 'rho 0.100'
num_classes = 3

dtws_to_compare = ['DTW_5', 'DTW_5_old']
sub_dirs = ['train', 'validation', 'test']

for sub_dir in sub_dirs:
    for i in range(num_classes):
        for file in glob(join(core_path, dataset_name, rho, dtws_to_compare[0], sub_dir, str(i+1), '*')):
            path = file
            path_to_compare = path.replace(dtws_to_compare[0], dtws_to_compare[1])

            file = Image.open(path)
            file = np.array(file.getdata())

            file_to_compare = Image.open(path_to_compare)
            file_to_compare = np.array(file_to_compare.getdata())

            assert np.isclose(file, file_to_compare).all()

print('The two datasets are equals')

