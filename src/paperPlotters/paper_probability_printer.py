import operator
from collections import Counter
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from dataset.files import TimeSeries, RefPattern
from utils.specification import specs

dataset = 'rational'
dataset_type = 'DTW'
window_size = 5
# rho_arr = ['0.001', '0.100', '0.500']
rho_arr = ['0.100']
num_sample = 7

length = specs[dataset]['x_dim']
y_dim = specs[dataset]['y_dim']
stream_id = 0

max_len = num_sample * length

if dataset == 'gunpoint':
    stream_name = f'STREAM_cycles-per-label-20_set-test_id-{stream_id}.npy'
else:
    stream_name = f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_' \
        f'set-test_id-{stream_id}.npy'

t = TimeSeries(
    f'../../data/{dataset}/{stream_name}')
timeseries = t.timeseries
total_len = len(timeseries)

network_type = '_ResNet'

full_pro = np.zeros((len(rho_arr)*2, total_len, y_dim))
starting_path = f'../../experiment_summaries/{dataset}/'

for idx, rho in enumerate(rho_arr):

    to_load_path = join(starting_path, f'rho {rho}', f'{dataset_type}_{window_size}',
                        'probabilities_value_not_shifted.npy')

    full_pro[idx, window_size:, :] = np.load(to_load_path)

    to_load_path = join(starting_path, f'rho {rho}', f'{dataset_type}_{window_size}{network_type}',
                        'probabilities_value_not_shifted.npy')
    full_pro[idx+1, window_size:, :] = np.load(to_load_path)


# START PLOTTING
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig = plt.figure(figsize=(13, 5.25), dpi=100)

plt.rcParams['font.family'] = "Gulasch", "Times", "Times New Roman", "serif"
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = 0.5 * plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = 0.5 * plt.rcParams['font.size']
# <<<<<<  ADD THIS IN  REMAING SCRIPTS use  r  before
plt.rcParams['text.usetex'] = True


gs = fig.add_gridspec(len(rho_arr) + 2, (len(rho_arr) + 2) * 2,
                      wspace=0.1, hspace=0.1,
                      height_ratios=None, width_ratios=None)

f_axi1 = fig.add_subplot(gs[0, :])
f_axi1.plot(timeseries[total_len - max_len:total_len], 'k', lw=1)
f_axi1.set_xlim(0, max_len)
f_axi1.axes.get_yaxis().set_visible(False)
f_axi1.axes.get_xaxis().set_visible(False)
f_axi1.set_title(r'STS')  # <<<<<<<<<<<<<<<<< ADD THIS

for i in range(0, num_sample):

    base_len = total_len - (num_sample * length)
    tmp_labels = dict(Counter(t.labels[base_len + (i * length):base_len + ((i + 1) * length)]))
    label = int(max(tmp_labels.items(), key=operator.itemgetter(1))[0])

    if i != 0:
        f_axi1.axvline(x=i * length, lw=0.7, color='k', linestyle='--')
    f_axi1.axvspan(i * length, (i + 1) * length,
                   facecolor=colors[label - 1], alpha=0.2)


for idx, rho in enumerate(rho_arr):

    #CNN
    # AXIS 2
    f_axi = fig.add_subplot(gs[idx + 1, :])
    for cla in range(full_pro.shape[2]):
        f_axi.plot(full_pro[idx, total_len - max_len:total_len, cla], colors[cla], alpha=0.7)

    rho_num = float(rho.split('_')[0])
    if len(rho.split('_')) > 1:
        base_patt = rho.split('_')[2]
    else:
        base_patt = ''

    if rho_num == 0.001:
        rho_num = 0.01

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<< I'VE CHANGED THIS. USE THE PAPER NOTATION. Real, A, B and AB
    # base_patt = rf'w={rho_num:.3f}'
    base_patt = 'CNN'
    if base_patt != '':
        text = rf'{base_patt}'
    else:
        text = r'Real'
    f_axi.text(
        0.005, 0.85, text,
        transform=f_axi.transAxes  # , bbox=props
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    f_axi.set_xlim(0, max_len)
    f_axi.axes.get_xaxis().set_visible(False)
    if rho != rho_arr[-1]:
        f_axi.axes.get_yaxis().set_visible(False)
    else:
        f_axi.axes.set_ylabel(r'Prob.')  # <<<<<<<< ADD THIS
        # f_axi.axes.set_xlabel(r'Class Pr.')  # <<<<<<<< ADD THIS

    for i in range(0, max_len, length):
        if i != 0:
            f_axi.axvline(x=i, lw=0.7, color='k', linestyle='--')

    # ResNet
    # AXIS 2
    idx = idx + 1
    f_axi = fig.add_subplot(gs[idx + 1, :])
    for cla in range(full_pro.shape[2]):
        f_axi.plot(full_pro[idx, total_len - max_len:total_len, cla], colors[cla], alpha=0.7)

    rho_num = float(rho.split('_')[0])
    if len(rho.split('_')) > 1:
        base_patt = rho.split('_')[2]
    else:
        base_patt = ''

    if rho_num == 0.001:
        rho_num = 0.01

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<< I'VE CHANGED THIS. USE THE PAPER NOTATION. Real, A, B and AB
    #base_patt = rf'w={rho_num:.3f}'
    base_patt = 'ResNet'
    if base_patt != '':
        text = rf'{base_patt}'
    else:
        text = r'Real'
    f_axi.text(
        0.005, 0.85, text,
        transform=f_axi.transAxes  # , bbox=props
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    f_axi.set_xlim(0, max_len)
    if rho != rho_arr[-1]:
        f_axi.axes.get_yaxis().set_visible(False)
    else:
        f_axi.axes.set_ylabel(r'Prob.')  # <<<<<<<< ADD THIS
        f_axi.axes.set_xlabel(r'Time')  # <<<<<<<< ADD THIS

    for i in range(0, max_len, length):
        if i != 0:
            f_axi.axvline(x=i, lw=0.7, color='k', linestyle='--')


plt.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.07)
plt.show()
# plt.savefig(f'{figure_name}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)