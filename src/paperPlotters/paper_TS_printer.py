import operator
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Patch

from dataset.dataset import Dataset
from dataset.files import TimeSeries, RefPattern, DTW
from utils.specification import specs, cmap

dataset_array = ['cbf', 'rational', 'gunpoint']
base_pattern = True
pattern_name = 'AB'
rho = '0.100'
num_sample = 15

fig = plt.figure(figsize=(13, 5.25), dpi=100)

plt.rcParams['font.family'] = "Gulasch", "Times", "Times New Roman", "serif"
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = 0.5 * plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = 0.5 * plt.rcParams['font.size']

gs = fig.add_gridspec(len(dataset_array), len(dataset_array)*2)
gs.update(wspace=0.100, hspace=0.4)

# START PLOTTING
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for idx, dataset in enumerate(dataset_array):
    dataset_name = dataset if not base_pattern else dataset+'_base'
    length = specs[dataset_name]['x_dim']
    y_dim = specs[dataset_name]['y_dim']
    stream_id = 0

    max_len = num_sample*length

    if dataset == 'gunpoint':
        ref_name = 'REF_num-5.npy'
        stream_name = f'STREAM_cycles-per-label-20_set-test_id-{stream_id}.npy'
    else:
        stream_name = f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_' \
                      f'set-test_id-{stream_id}.npy'
        ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'

    if base_pattern:
        ref_name = f'BASE_REF_len-{length}_noise-5_num-1.npy'
        if len(pattern_name) > 0:
            ref_name = f'BASE_REF_len-{length}_noise-5_num-1_{pattern_name}.npy'

    t = TimeSeries(
        f'../data/{dataset}/{stream_name}')
    timeseries = t.timeseries

    total_len = len(timeseries)

    f_axi1 = fig.add_subplot(gs[idx, :])
    f_axi1.set_title(f'{"CBF" if dataset == "cbf" else dataset.capitalize()}')
    f_axi1.plot(timeseries[total_len-max_len:total_len], 'k', lw=1)
    f_axi1.set_xlim(0, max_len)
    f_axi1.axes.get_yaxis().set_visible(False)
    f_axi1.set_xticks(range(0, max_len+length, length))
    # f_axi1.axes.get_xaxis().set_visible(False)

    for i in range(0, num_sample):

        base_len = total_len-(num_sample*length)
        tmp_labels = dict(Counter(t.labels[base_len+(i*length):base_len+((i+1)*length)]))
        label = int(max(tmp_labels.items(), key=operator.itemgetter(1))[0])

        if i != 0:
            f_axi1.axvline(x=i*length, lw=0.7, color='k', linestyle='--')
        f_axi1.axvspan(i*length, (i + 1)*length, facecolor=colors[label-1], alpha=0.2)


legend_elements = []
for i in range(4):
    legend_elements.append(Patch(facecolor=colors[i], alpha=0.2,
                           label=f'Class #{i+1}'))

plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.10),
           fancybox=False, shadow=True, ncol=4)
plt.show()
