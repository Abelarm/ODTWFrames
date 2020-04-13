import operator
from glob import glob

from os.path import join
from collections import Counter

import numpy as np


class ProtoTimeSeries:

    def get_properties(self):
        """
        Returns the property of the generated time series

        :return: dictionary of properties
        """
        tmp_path = self.path.split(f'/{self.ts_type}_')[1].replace('.npy', '')

        keys, values = [], []
        for split in tmp_path.split('_'):

            *k, v = split.split('-')
            if len(k) > 1:
                k = '_'.join(k)
            else:
                k = k[0]
            keys.append(k)
            values.append(v)

        return {k: v for k, v in zip(keys, values)}


class TimeSeries(ProtoTimeSeries):
    path = None
    timeseries = None
    labels = None
    ts_type = 'STREAM'

    def __init__(self, path):
        tmp = np.load(path)
        self.timeseries = tmp[:, 0]
        self.labels = tmp[:, 1]
        self.path = path


class RefPattern(ProtoTimeSeries):
    path = None
    ts_type = 'REF'

    def __init__(self, path):
        patterns = np.load(path)

        self.lab_patterns = []
        for pattern in patterns:
            tmp_dict = {'label': pattern[0],
                        'pattern': pattern[1:]}
            self.lab_patterns.append(tmp_dict)


class DTW:
    ref_pattern = None
    time_series = None
    class_num = None
    path = None
    dtw = None
    from_pattern_idx = None
    selected = None

    def __init__(self, ref_pattern, time_series, class_num, rho, starting_path, ref_id=None):
        self.ref_pattern = ref_pattern
        self.time_series = time_series
        self.class_num = class_num
        self.starting_skip = 0
        self.from_pattern_idx = ref_id

        # Find the DTW associated to the ref_pattern, timeseries and class_num
        time_series_prop = time_series.get_properties()

        labels = [lb['label'] for lb in self.ref_pattern.lab_patterns]

        if time_series_prop['set'] not in ['train', 'validation'] and self.from_pattern_idx is None:

            selectable_labels = np.argwhere(np.array(labels) == class_num)
            selectable_labels = selectable_labels.reshape(-1)
            selected = np.random.choice(selectable_labels)

        elif self.from_pattern_idx is not None:
            selected = self.from_pattern_idx
        else:
            if 'gunpoint' in starting_path:
                path = f'dtwMat-{time_series_prop["set"]}_rho-{rho}_ref-id-*_stream-id-{time_series_prop["id"]}.npy'
            else:
                path = f'dtwMat-{time_series_prop["set"]}_length-{time_series_prop["length"]}_' \
                       f'noise-{time_series_prop["noise"]}_warp-{time_series_prop["warp"]}' \
                       f'_shift-{time_series_prop["shift"]}_outliers-{time_series_prop["outliers"]}' \
                       f'_rho-{rho}_ref-id-*_stream-id-{time_series_prop["id"]}.npy'

            a = glob(join(starting_path, path))
            ids = [int(f.split('ref-id-')[1].split('_')[0]) for f in a]
            for indx in ids:
                if labels[indx] == class_num:
                    selected = indx
                    break

        print(f'selected index {selected}')
        self.from_pattern_idx = selected

        if 'gunpoint' in starting_path:
            self.path = f'dtwMat-{time_series_prop["set"]}_rho-{rho}_ref-id-{selected}_stream-id-{time_series_prop["id"]}.npy'
        else:
            self.path = f'dtwMat-{time_series_prop["set"]}_length-{time_series_prop["length"]}_' \
                   f'noise-{time_series_prop["noise"]}_warp-{time_series_prop["warp"]}' \
                   f'_shift-{time_series_prop["shift"]}_outliers-{time_series_prop["outliers"]}' \
                   f'_rho-{rho}_ref-id-{selected}_stream-id-{time_series_prop["id"]}.npy'

        self.dtw = np.load(join(starting_path, self.path))
        print(f'loaded the file {self.path}')

        if not ref_id:
            self.selected = selected

    def images(self, window_size):

        images = []
        labels = []
        for i in range(self.starting_skip, self.dtw.shape[1] - window_size):
            tmp_img = self.dtw[:, i:i + window_size]
            tmp_labels = dict(Counter(self.time_series.labels[i:i + window_size]))

            max_val = max(tmp_labels.items(), key=operator.itemgetter(1))
            if len([k for k, v in tmp_labels.items() if v == max_val[1]]) > 1:
                print(f'Skipped the frame from {i}:{i + window_size} due to not trivial label')
                continue

            images.append(tmp_img)
            labels.append(int(max_val[0]))

        return images, labels
