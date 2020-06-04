import operator
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import transforms

from dataset.dataset import Dataset
from dataset.files import TimeSeries, RefPattern, DTW
from utils.specification import specs, cmap

dataset = 'cbf'
base_pattern = True
pattern_name = 'AB'
rho = '0.100'
num_sample = 7

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

ref = RefPattern(f'../data/{dataset}/{ref_name}')

ref_ids = specs[dataset_name]['ref_id']
if pattern_name == 'FULL':
    ref_ids = range(5)
elif len(pattern_name) > 0:
    ref_ids = range(len(pattern_name))

if base_pattern:
    rho_name = f'rho {rho}_base'
    if len(pattern_name) > 0:
        rho_name = f'rho {rho}_base_{pattern_name}'
else:
    rho_name = f'rho {rho}'

dtws = []
for idx, ref_id in enumerate(ref_ids):
    dtw = DTW(ref, t, class_num=idx+1, rho=f'{rho}',
              starting_path=f'../data/{dataset}/{rho_name}',
              ref_id=ref_id)
    dtws.append(dtw)

window_size = 25
Dataset.image_creator(*dtws, window_size=window_size)

# START PLOTTING
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

total_len = len(timeseries)

fig = plt.figure(figsize=(13, 5.25), dpi=100)

plt.rcParams['font.family'] = "Gulasch", "Times", "Times New Roman", "serif"
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = 0.5 * plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = 0.5 * plt.rcParams['font.size']

gs = fig.add_gridspec(len(dtws)+1, (len(dtws)+1)*2,
                      wspace=0.1, hspace=0.1,
                      height_ratios=None, width_ratios=None)

f_axi1 = fig.add_subplot(gs[0, 1:])
f_axi1.plot(timeseries[total_len-max_len:total_len], 'k', lw=1)
f_axi1.axes.get_yaxis().set_visible(False)
f_axi1.axes.get_xaxis().set_visible(False)

for i in range(0, num_sample):

    base_len = total_len-(num_sample*length)
    tmp_labels = dict(Counter(t.labels[base_len+(i*length):base_len+((i+1)*length)]))
    label = int(max(tmp_labels.items(), key=operator.itemgetter(1))[0])

    if i != 0:
        f_axi1.axvline(x=i*length, lw=0.7, color='k', linestyle='--')
    f_axi1.axvspan(i*length, (i + 1)*length, facecolor=colors[label-1], alpha=0.2)


for idx, dtw in enumerate(dtws):
    # AXIS 2
    f_axi = fig.add_subplot(gs[idx+1, 1:], sharex=f_axi1)
    img = f_axi.imshow(dtw.img[:, total_len-max_len:total_len], cmap=cmap, origin='lower', aspect='auto')
    f_axi.axes.get_yaxis().set_visible(False)
    if idx != len(dtws)-1:
        f_axi.axes.get_xaxis().set_visible(False)

    for i in range(0, max_len, length):
        if i != 0:
            f_axi.axvline(x=i, lw=0.7, color='k', linestyle='--')


    # SUB AXIS 2,1
    f_axi_1 = fig.add_subplot(gs[idx+1, 0
                              ])
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
    if base_pattern:
        color_idx = idx + y_dim
    else:
        color_idx = idx
    f_axi_1.plot(ref.lab_patterns[dtw.from_pattern_idx]['pattern'], color=colors[color_idx], transform=rot + base)
    if base_pattern and dtw.from_pattern_idx == 0:
        val_min = ref.lab_patterns[dtw.from_pattern_idx+1]['pattern'].min()
        val_max = ref.lab_patterns[dtw.from_pattern_idx+1]['pattern'].max()
        f_axi_1.set_xlim(-val_max, -val_min)
    f_axi_1.set_yticks([0, 100])
    if idx != len(dtws)-1:
        f_axi_1.set_xticks([])
        f_axi_1.set_yticks([])

    # fig.colorbar(img, cmap=cmap, ax=f_axi)

figure_name = f'{dataset}_{"universal" if base_pattern else "real"}'
plt.savefig(f'{figure_name}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
