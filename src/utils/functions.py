import re
from weakref import WeakValueDictionary


def get_id_interval(filename):
    # print(filename)

    if '.png' in filename:
        filename = filename.replace('.png', '')

        ts_id = re.search('[0-9]+/(.+)_', filename, re.IGNORECASE).group(1)
        interval = re.search('[0-9]+_(.*)', filename, re.IGNORECASE).group(1)

    else:
        filename = filename.replace('.npy', '')

        ts_id = re.search(':(.*)_', filename, re.IGNORECASE).group(1)
        interval = re.search('_(.*)\\|', filename, re.IGNORECASE).group(1)

    interval = list(map(int, interval.split('-')))

    return ts_id, interval


class Singleton(type):
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # This variable declaration is required to force a
            # strong reference on the instance.
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Paths(metaclass=Singleton):

    dataset = None
    dataset_type = None
    rho = None
    window_size = None
    base_pattern = None
    pattern_name = None
    core_path = None

    rho_name = ''
    model_name = ''

    def __init__(self, dataset, dataset_type, rho, window_size, base_pattern, pattern_name, column_scale=False,
                 core_path='../../..'):

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.rho = rho
        self.window_size = window_size
        self.base_pattern = base_pattern
        self.pattern_name = pattern_name
        self.column_scale = column_scale
        self.core_path = core_path

        self.get_rho_name()
        self.get_model_name()

    def get_rho_name(self):

        rho_name = ''

        if self.rho:
            rho_name = f'rho {self.rho}'
        if self.dataset_type == 'RP':
            rho_name = 'RP'

        if self.base_pattern:
            rho_name += f'_base_{self.pattern_name}'

        self.rho_name = rho_name
        return rho_name

    def get_model_name(self):
        self.model_name = f'{self.dataset_type}_{self.window_size}'

        if self.column_scale:
            self.model_name += '_column_scale'

        return self.model_name

    def get_beginning_path(self):

        return f'{self.core_path}/data/{self.dataset}'

    def get_dtw_path(self):

        beginning_path = self.get_beginning_path()

        dtw_path = f'{beginning_path}/{self.rho_name}'
        return dtw_path

    def get_data_path(self):

        beginning_path = self.get_beginning_path()

        data_path = f'{beginning_path}/{self.rho_name}/{self.model_name}'
        data_path = data_path.replace('_column_scale', '')

        return data_path

    def get_weight_dir(self):

        if not self.rho:
            weight_dir = f'{self.core_path}/Network_weights/{self.dataset}/{self.dataset_type}'
        else:
            weight_dir = f'{self.core_path}/Network_weights/{self.dataset}/{self.rho_name}'

        return weight_dir

    def get_summaries_path(self):

        path_dir = f'{self.core_path}/experiment_summaries/{self.dataset}/{self.rho_name}/{self.model_name}'

        return path_dir

    def get_error_path(self):

        path_dir = f'{self.core_path}/error_analysis/{self.dataset}/{self.rho_name}/{self.model_name}'

        return path_dir

    def get_explain_path(self):

        path_dir = f'{self.core_path}/Network_explain/{self.dataset}/{self.rho_name}/{self.model_name}'
        return path_dir