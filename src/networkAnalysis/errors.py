import re
from collections import Counter
from glob import glob
from os import makedirs
from os.path import isdir, join

from keras_preprocessing.image import DirectoryIterator
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from dataset.files import TimeSeries

core_path = '../../..'


def analysis(X_generator, Y_true, Y_pred, save_dir, dataset_name):

    if not isdir(save_dir) or not isdir(join(save_dir, 'wrong_samples')):
        makedirs(save_dir, exist_ok=True)
        makedirs(join(save_dir, 'wrong_samples'), exist_ok=True)

    _accuracy_by_innerposition(X_generator, Y_true, Y_pred, save_dir, dataset_name=dataset_name)
    _calculate_acc_by_dist(X_generator, Y_true, Y_pred, save_dir, dataset_name=dataset_name)

    max_images = 200
    printed_images = 0
    pred_argm = np.argmax(Y_pred, axis=1)
    for i in tqdm(range(len(X_generator))):

        if type(X_generator) is DirectoryIterator:
            true = Y_true[i] + 1
        else:
            true = Y_true[i]

        pred = pred_argm[i]

        if true != pred:
            printed_images += 1
            _wrong_sample(X_generator[i],  X_generator.filenames[i], true, pred,
                          join(save_dir, 'wrong_samples'),
                          dataset_name)

        if printed_images > max_images:
            break


def _calculate_acc_by_dist(X_generator, Y_true, Y_pred, save_dir, dataset_name):

    accuracy_result = dict()
    accuracy_result[0] = []
    for i in tqdm(range(len(X_generator))):

        true = Y_true[i]
        pred = Y_pred[i]

        filename = X_generator.filenames[i]
        if '.png' in filename:
            filename = filename.replace('.png', '')
            # print(filename)
            ts_id, interval = filename.split("/")[1].split("_")
            interval = list(map(int, interval.split('-')))
        else:
            filename = filename.replace('.npy', '')
            # print(filename)
            ts_id = re.search(':(.*)_', filename, re.IGNORECASE).group(1)
            interval = re.search('_(.*)\\|', filename, re.IGNORECASE).group(1)
            interval = list(map(int, interval.split('-')))

        file_name = f'{core_path}/data/{dataset_name}/' \
                    f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set' \
                    f'-test_id-{ts_id}.npy'

        if dataset_name == 'gunpoint':
            file_name = f'{core_path}/data/{dataset_name}/' \
                        f'STREAM_cycles-per-label-20_set-test_id-{ts_id}.npy'

        t = TimeSeries(file_name)
        labels = t.labels[interval[0]:interval[1]]
        true_hot = X_generator[i][1]

        curr_accuracy = 1 - cosine(true_hot, pred)

        if type(X_generator) is DirectoryIterator:

            if type(true) is list or type(true) is np.ndarray:
                curr_y_arg = true[0] + 1
            else:
                curr_y_arg = true + 1
        else:
            if type(true) is list or type(true) is np.ndarray:
                curr_y_arg = true[0]
            else:
                curr_y_arg = true

        count_lab = dict(Counter(labels))
        if len(count_lab) == 1:
            accuracy_result[0].append(curr_accuracy)
        else:
            count_other = 0
            for k in count_lab:
                if k != curr_y_arg:
                    count_other += count_lab[k]
                    break

            if count_other not in accuracy_result:
                accuracy_result[count_other] = [curr_accuracy]
            else:
                accuracy_result[count_other].append(curr_accuracy)

    accuracy_result = dict((k, np.mean(v)) for k, v in accuracy_result.items())
    ordered = {k: v for k, v in sorted(accuracy_result.items(), key=lambda item: item[0])}
    plt.figure(dpi=300)
    plt.plot(list(map(int, ordered.keys())), list(ordered.values()), 'bo-')

    for x, y in zip(list(map(int, ordered.keys())), list(ordered.values())):
        label = f"{y:.2f}"

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 5),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    plt.title('Accuracy in function of number other label')
    plt.savefig(join(save_dir, 'accuracy_by_distance.pdf'))
    print(accuracy_result)


def _wrong_sample(X, filename, true, pred, save_dir, dataset_name):

    if '.png' in filename:
        filename = filename.replace('.png', '')
        # print(filename)
        ts_id, interval = filename.split("/")[1].split("_")
        interval = list(map(int, interval.split('-')))
    else:
        filename = filename.replace('.npy', '')
        # print(filename)
        ts_id = re.search(':(.*)_', filename, re.IGNORECASE).group(1)
        interval = re.search('_(.*)\\|', filename, re.IGNORECASE).group(1)
        interval = list(map(int, interval.split('-')))

    file_name = f'{core_path}/data/{dataset_name}/' \
                f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set' \
                f'-test_id-{ts_id}.npy' \

    if dataset_name == 'gunpoint':
            file_name = f'{core_path}/data/{dataset_name}/' \
                        f'STREAM_cycles-per-label-20_set-test_id-{ts_id}.npy'

    t = TimeSeries(file_name)
    image = X[0][0]

    image = (image-image.min())/(image.max()-image.min())

    if len(image.shape) == 3 and 5 > image.shape[-1] > 2:

        fig = plt.figure(constrained_layout=False, dpi=300)
        gs = fig.add_gridspec(2, 12)
        fig.suptitle(f'True label: {true} - Predicted: {pred}')

        extra_window = image.shape[1]
        f_axi1 = fig.add_subplot(gs[0, :])
        f_axi1.plot(t.timeseries[interval[0] - extra_window:interval[1] + extra_window])
        f_axi1.plot(t.labels[interval[0] - extra_window:interval[1] + extra_window])
        len_series = t.labels[interval[0] - extra_window:interval[1] + extra_window].shape[0]
        f_axi1.axis(xmin=0, xmax=len_series)
        f_axi1.axvline(x=extra_window, linewidth=2, color='r')
        f_axi1.axvline(x=extra_window * 2, linewidth=2, color='r')

        f_axi2 = fig.add_subplot(gs[1, 4:8])
        f_axi2.axis('off')
        f_axi2.imshow(image, origin='lower', aspect='auto')

    else:

        true = true[0]

        fig = plt.figure(constrained_layout=False, dpi=300)
        fig.suptitle(f'True label: {true} - Predicted: {pred}')

        if len(image.shape) == 3:
            extra_window = image.shape[1]
        else:
            extra_window = image.shape[0]
        plt.plot(t.timeseries[interval[0] - extra_window:interval[1] + extra_window])
        plt.plot(t.labels[interval[0] - extra_window:interval[1] + extra_window])
        len_series = len(t.labels[interval[0] - extra_window:interval[1] + extra_window])
        plt.xlim(0, len_series)
        plt.axvline(x=extra_window, linewidth=2, color='r')
        plt.axvline(x=extra_window * 2, linewidth=2, color='r')

    save_path = join(save_dir, f'TS:{ts_id}###True:{true}-Pred:{pred}###{interval[0]}-{interval[1]}.pdf')
    plt.savefig(save_path)
    plt.close(fig)


def _accuracy_by_innerposition(X_generator, Y_true, Y_pred, save_dir, dataset_name):
    accuracy_result = dict()

    for i in tqdm(range(len(X_generator))):

        true = Y_true[i]
        pred = Y_pred[i]

        filename = X_generator.filenames[i]
        if '.png' in filename:
            filename = filename.replace('.png', '')
            # print(filename)
            ts_id, interval = filename.split("/")[1].split("_")
            interval = list(map(int, interval.split('-')))
        else:
            filename = filename.replace('.npy', '')
            # print(filename)
            ts_id = re.search(':(.*)_', filename, re.IGNORECASE).group(1)
            interval = re.search('_(.*)\\|', filename, re.IGNORECASE).group(1)
            interval = list(map(int, interval.split('-')))

        true_hot = X_generator[i][1]
        curr_accuracy = 1 - cosine(true_hot, pred)

        if dataset_name == 'gunpoint':

            val = interval[1] % 150

            if val < 50:
                where = 'start'
            elif val < 100:
                where = 'middle'
            else:
                where = 'end'
        else:
            tent_last = str(interval[1])
            tent_last = int(tent_last[-2:])

            if tent_last < 33:
                where = 'start'
            elif tent_last < 66:
                where = 'middle'
            else:
                where = 'end'

        if where not in accuracy_result:
            accuracy_result[where] = [curr_accuracy]
        else:
            accuracy_result[where].append(curr_accuracy)

    accuracy_result = dict((k, np.mean(v)) for k, v in accuracy_result.items())
    values_to_plot = [accuracy_result['start'], accuracy_result['middle'], accuracy_result['end']]
    labels = ['start', 'middle', 'end']
    plt.figure(dpi=300)
    plt.plot(range(len(labels)), values_to_plot, 'bo-')
    plt.xticks(range(len(labels)), labels, rotation='horizontal')

    for x, y in zip(list(range(len(labels))), values_to_plot):
        label = f"{y:.3f}"

        plt.annotate(label,  # this is the text
                     (x, y),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 5),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    plt.title('Accuracy in function of inner distance')
    plt.savefig(join(save_dir, 'accuracy_by_innerdistance.pdf'))
    print(accuracy_result)
