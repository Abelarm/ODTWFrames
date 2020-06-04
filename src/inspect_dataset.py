import matplotlib.pyplot as plt
from matplotlib import transforms

from dataset.dataset import Dataset
from dataset.files import TimeSeries, RefPattern, DTW, RP
from utils.functions import Paths
from utils.specification import specs, cmap

dataset = 'rational'
base_pattern = False
pattern_name = ''
rho = '0.100'
dataset_type = 'RP'

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

if dataset_type == 'DTW':
    image_class = DTW
elif dataset_type == 'RP':
    image_class = RP
    rho_name = 'RP'

dataset_name = dataset if not base_pattern else dataset + '_base'
length = specs[dataset_name]['x_dim']
stream_id = 0

paths = Paths(dataset, dataset_type, rho, 5, base_pattern, pattern_name, core_path='../')
beginning_path = paths.get_beginning_path()
data_path = paths.get_data_path()

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

t = TimeSeries(f'{beginning_path}/{stream_name}')
timeseries = t.timeseries

ref = RefPattern(f'{beginning_path}/{ref_name}')

ref_ids = specs[dataset_name]['ref_id']


imgs = []
for idx, ref_id in enumerate(ref_ids):
    dtw = image_class(ref, t, class_num=idx + 1, rho=f'{rho}',
                      starting_path=paths.get_dtw_path(),
                      ref_id=ref_id)
    imgs.append(dtw)

window_size = 25
Dataset.image_creator(*imgs, window_size=window_size)

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(len(imgs) + 1, 12)

f_axi1 = fig.add_subplot(gs[0, 1:])
f_axi1.plot(timeseries, label='val')
f_axi1.plot(t.labels, label='lab')
f_axi1.axes.get_yaxis().set_visible(False)
f_axi1.axes.get_xaxis().set_visible(False)

for idx, img_class in enumerate(imgs):
    # AXIS 2
    f_axi = fig.add_subplot(gs[idx + 1, 1:], sharex=f_axi1)
    img = f_axi.imshow(img_class.img, cmap=cmap, origin='lower', aspect='auto')
    f_axi.axes.get_yaxis().set_visible(False)
    if idx != len(imgs) - 1:
        f_axi.axes.get_xaxis().set_visible(False)

    # SUB AXIS 2,1
    f_axi_1 = fig.add_subplot(gs[idx + 1, 0])
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
    f_axi_1.plot(ref.lab_patterns[img_class.from_pattern_idx]['pattern'], transform=rot + base)
    if base_pattern and img_class.from_pattern_idx == 0:
        val_min = ref.lab_patterns[img_class.from_pattern_idx]['pattern'].min()
        val_max = ref.lab_patterns[img_class.from_pattern_idx]['pattern'].max()
        f_axi_1.set_xlim(-val_max * 2, val_max * 2)
    f_axi_1.axis('off')

    fig.colorbar(img, cmap=cmap, ax=f_axi)

# fig.subplots_adjust(right=0.90)
# cbar_ax = fig.add_axes([0.91, 0.12, 0.01, 0.64])
# fig.colorbar(img, cmap='plasma', cax=cbar_ax)

plt.show()
