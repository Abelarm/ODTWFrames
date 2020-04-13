import operator
import re
from collections import Counter
from glob import glob
from os.path import join
from random import choice

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.preprocessing import OneHotEncoder

from dataset.files import TimeSeries

dpi = 72
core_path = '../../..'

def plot_acc_loss(history, epochs, save_dir):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(join(save_dir, 'acc_loss.pdf'))


def plot_roc_auc(n_classes, y_true, y_pred, save_dir):

    enc = OneHotEncoder(sparse=False)
    y_true = enc.fit_transform(y_true.reshape(-1, 1))

    fig = plt.figure(dpi=dpi)

    if y_pred.shape[1] > 2:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f"micro-average ROC curve (area = {roc_auc['micro']:0.2f})",
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f"micro-average ROC curve (area = {roc_auc['macro']:0.2f})",
                 color='navy', linestyle=':', linewidth=4)

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f"ROC curve of class {i} (area = {roc_auc[i]:0.2f})")
        plt.title('Some extension of Receiver operating characteristic to multi-class')
    else:
        selected_class = 1
        fpr, tpr, _ = roc_curve(y_true[:, selected_class], y_pred[:, selected_class])
        auc_area = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve calculated on class {selected_class} (area = {auc_area:0.2f})')

        plt.title('Receiver operating two-class')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(join(save_dir, 'ROC_AUC.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):

    y_pred = np.argmax(y_pred, axis=1)

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]

    for title, normalize in titles_options:
        fig = plt.figure(dpi=dpi)
        conf = confusion_matrix(y_true, y_pred, normalize=normalize)

        con_mat_df = pd.DataFrame(conf, index=class_names, columns=class_names)
        sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if normalize:
            plt.title('Confusion matrix normalized')
            plt.savefig(join(save_dir, 'Confusion matrix normalized.pdf'))
            plt.close(fig)
        else:
            plt.title('Confusion matrix')
            plt.savefig(join(save_dir, 'Confusion matrix.pdf'), bbox_inches='tight')
            plt.close(fig)


def plot_class_probabilities(X_generator, y_dim, dataset_name, rho, model_name, window_size, y_pred,  save_dir):

    file_name = f'{core_path}/data/{dataset_name}/' \
                f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set' \
                f'-test_id-*.npy'

    if dataset_name == 'gunpoint':
        file_name = f'{core_path}/data/{dataset_name}/' \
                  f'STREAM_cycles-per-label-20_set-test_id-*.npy'

    list = glob(file_name)

    # select random stream
    t_name = choice(list)
    ts_id = re.search('test_id-(.*).npy', t_name, re.IGNORECASE).group(1)
    t = TimeSeries(t_name)
    timeseries = t.timeseries
    labels = t.labels

    probabilities_value = np.empty((timeseries.shape[0]-window_size, y_dim))
    prediction_value = np.zeros(timeseries.shape[0])

    x_dim = X_generator.__getitem__(0)[0].shape
    print(x_dim)

    for i in range(0, timeseries.shape[0] - window_size):

        interval = f'{i}-{i+window_size}'

        tmp_labels = dict(Counter(labels[i:i + window_size]))

        label = int(max(tmp_labels.items(), key=operator.itemgetter(1))[0])

        file = glob(f'{core_path}/data/{dataset_name}/rho {rho}/{model_name}/test/{label}/{ts_id}_{interval}.png')
        if len(file) == 1:
            filename = f'{label}/{ts_id}_{interval}.png'
        else:
            file = glob(f'{core_path}/data/{dataset_name}/{model_name}/test/X:{ts_id}_{interval}|Y:{label}.npy')
            if len(file) == 1:
                filename = f'X:{ts_id}_{interval}|Y:{label}.npy'
            else:
                file = glob(f'{core_path}/data/{dataset_name}/rho {rho}/{model_name}/test/X:{ts_id}_{interval}|Y:{label}.npy')
                if len(file) == 1:
                    filename = f'X:{ts_id}_{interval}|Y:{label}.npy'
                else:
                    raise FileNotFoundError("DIDN'T FIND THE FILE .npy")

        index_file = X_generator.filenames.index(filename)
        probabilities_value[i] = y_pred[index_file]
        prediction_value[i+window_size] = 1 if np.argmax(y_pred[index_file]) == (label-1) else -1

    fig = plt.figure(constrained_layout=False, dpi=dpi)

    plt.subplot(y_dim+1, 1, 1)
    plt.title('Probability per class')
    plt.plot(t.timeseries, 'k', lw=0.7)
    plt.plot(prediction_value, 'red', lw=0.5, alpha=0.6)
    plt.xlim(0, timeseries.shape[0])
    plt.yticks([])
    plt.xticks([])
    if dataset_name == 'gunpoint':
        class_len = 150
    else:
        class_len = 100
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range(0, timeseries.shape[0], class_len):
        plt.axvline(x=i, lw=0.5, color='k', linestyle='--')
        
        tmp_labels = dict(Counter(labels[i:i + class_len]))
        label = int(max(tmp_labels.items(), key=operator.itemgetter(1))[0])

        plt.axvspan(i, i + class_len, facecolor=colors[label-1], alpha=0.2)

    for i in range(y_dim):

        plt.subplot(y_dim+1, 1, i+2)
        shifted_probabilities = np.append(np.zeros((window_size, 1)), probabilities_value[:, i])
        plt.plot(shifted_probabilities, colors[i], linewidth=0.6)

        for j in range(0, timeseries.shape[0], class_len):
            plt.axvline(x=j, lw=0.5, color='k', linestyle='--')

        plt.xlim(0, timeseries.shape[0])
        y_label = f'Class {i+1}'
        plt.ylabel(y_label)
        plt.yticks([])
        plt.xticks([])

    plt.savefig(join(save_dir, 'probability_per_class.pdf'), bbox_inches='tight')
    plt.close(fig)


