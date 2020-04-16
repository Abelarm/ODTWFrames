import operator
import os
from collections import Counter
from glob import glob
from os.path import join, isdir

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset.files import RefPattern, TimeSeries, DTW


class Dataset:
    ref_path = None
    reference = None
    stream_path = None
    starting_path = None
    rho = None
    window_size = None
    classes = None

    def __init__(self, ref_path, stream_path, stream_set, starting_path, rho, window_size, classes, max_id):
        self.ref_path = ref_path
        self.stream_path = stream_path

        self.stream_set = stream_set

        self.starting_path = starting_path
        self.rho = rho
        self.window_size = window_size
        self.classes = classes
        self.max_id = max_id

        self.reference = RefPattern(self.ref_path)

    def create_image_dataset(self, save_path, ref_ids=None, base_pattern=False):

        if not ref_ids:
            selected_ids = []

        for stream in tqdm(glob(self.stream_path)):
            dtws_tmp = []
            print(f'\n==== Computing images for file {stream} ====\n')
            id_path = stream.split('_id-')[1].split('.npy')[0]

            if int(id_path) > self.max_id:
                continue

            t = TimeSeries(stream)
            if not base_pattern:
                channel_iterator = self.classes
            else:
                channel_iterator = [int(x['label']) for x in self.reference.lab_patterns]
            if ref_ids:
                for c, ref_id in zip(channel_iterator, ref_ids):
                    dtws_tmp.append(DTW(self.reference,
                                        t,
                                        class_num=c,
                                        rho=self.rho,
                                        starting_path=self.starting_path,
                                        ref_id=ref_id))
            else:
                for c in channel_iterator:
                    dtw_tmp = DTW(self.reference,
                                  t,
                                  class_num=c,
                                  rho=self.rho,
                                  starting_path=self.starting_path)
                    dtws_tmp.append(dtw_tmp)

                    if dtw_tmp.selected not in selected_ids:
                        selected_ids.append(dtw_tmp.selected)

            images, labels = self.image_creator(*dtws_tmp, window_size=self.window_size)

            for i, (v, l) in enumerate(zip(images, labels)):

                if v.shape[-1] < 3 or v.shape[-1] > 4:
                    final_path = join(save_path, f'{self.stream_set}')
                    if not isdir(final_path):
                        os.makedirs(final_path)
                    np.save(join(final_path, f'X:{id_path}_{i}-{i + self.window_size}|Y:{int(l)}'), v)
                else:
                    final_path = join(save_path, f'{self.stream_set}/{l}')
                    if not isdir(final_path):
                        os.makedirs(final_path)

                    rescaled = (255.0 / v.max() * (v - v.min())).astype(np.uint8)
                    im = Image.fromarray(rescaled)
                    im.save(join(final_path, f'{id_path}_{i}-{i + self.window_size}.png'))

        if not ref_ids:
            return selected_ids

    def create_series_dataset(self, save_path):

        if not isdir(save_path):
            os.makedirs(save_path)

        for stream in tqdm(glob(self.stream_path)):
            print(f'\n==== Computing images for file {stream} ====\n')
            id_path = stream.split('_id-')[1].split('.npy')[0].replace('.npy', '')
            t = TimeSeries(stream)
            for i in range(0, t.timeseries.shape[0] - self.window_size):
                x = t.timeseries[i:i + self.window_size]
                x = x.reshape(self.window_size, -1)
                y = t.labels[i:i + self.window_size]

                tmp_labels = dict(Counter(y))

                max_val = max(tmp_labels.items(), key=operator.itemgetter(1))
                if len([k for k, v in tmp_labels.items() if v == max_val[1]]) > 1:
                    print(f'Skipped the frame from {i}:{i + self.window_size} due to not trivial label')
                    continue

                np.save(join(save_path, f'X:{id_path}_{i}-{i + self.window_size}|Y:{int(max_val[0])}'), x)

    def create_series_image_dataset(self, save_path, len_size=3, ref_ids=None):

        if not isdir(save_path):
            os.makedirs(save_path)

        for stream in tqdm(glob(self.stream_path)):
            dtws_tmp = []
            print(f'\n==== Computing images for file {stream} ====\n')
            id_path = stream.split('_id-')[1].split('.npy')[0].replace('.npy', '')
            t = TimeSeries(stream)
            if ref_ids:
                for c, ref_id in zip(self.classes, ref_ids):
                    dtws_tmp.append(DTW(self.reference,
                                        t,
                                        class_num=c,
                                        rho=self.rho,
                                        starting_path=self.starting_path,
                                        ref_id=ref_id))
            else:
                for c in self.classes:
                    dtws_tmp.append(DTW(self.reference,
                                        t,
                                        class_num=c,
                                        rho=self.rho,
                                        starting_path=self.starting_path))

            images, labels = self.image_creator(*dtws_tmp, window_size=self.window_size)

            for i in range(0, images.shape[0] - len_size):

                imgs = images[i:i + len_size, :, :]
                labs = labels[i:i + len_size]

                tmp_labels = dict(Counter(labs))

                max_val = max(tmp_labels.items(), key=operator.itemgetter(1))
                if len([k for k, v in tmp_labels.items() if v == max_val[1]]) > 1:
                    print(f'Skipped the frame from {i}:{i + self.window_size} due to not trivial label')
                    continue

                rescaled = (imgs - np.mean(imgs)) / np.std(imgs)
                final_path = join(save_path, f'{self.stream_set}')
                if not isdir(final_path):
                    os.makedirs(final_path)
                np.save(join(final_path, f'X:{id_path}_{i}-{i + len_size}|Y:{int(max_val[0])}'), rescaled)

    @staticmethod
    def image_creator(*dtws, window_size=25):

        imgs, labels = [], []
        for dtw in dtws:
            i, l = dtw.images(window_size)
            imgs.append(i)
            labels.append(l)

        labels_np = np.array(labels)[0, :]

        imgs_np = np.array(imgs)
        imgs_np = np.stack(imgs_np, axis=3)

        return imgs_np, labels_np
