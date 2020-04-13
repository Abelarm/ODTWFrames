import matplotlib.pyplot as plt
from matplotlib import transforms

from dataset.dataset import Dataset
from dataset.files import TimeSeries, RefPattern, DTW

dataset = 'cbf'
base_pattern = True
rho = '0.100'

if dataset == 'gunpoint':
    ref_name = 'REF_num-5.npy'
    stream_name = 'STREAM_cycles-per-label-20_set-test_id-0.npy'
elif base_pattern:
    ref_name = 'BASE_REF_len-100_noise-5_num-1.npy'
    stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-0.npy'
else:
    ref_name = 'REF_length-100_noise-5_warp-10_shift-10_outliers-0_num-10.npy'
    stream_name = 'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set-test_id-0.npy'

t = TimeSeries(
    f'../data/{dataset}/{stream_name}')
timeseries = t.timeseries

ref = RefPattern(f'../data/{dataset}/{ref_name}')


ref_ids = [0, 1, 2, 3, 4, 5, 6]

dtws = []
for idx, ref_id in enumerate(ref_ids):
    if base_pattern:
        dtw = DTW(ref, t, class_num=idx + 1, rho=f'{rho}',
                  starting_path=f'../data/{dataset}/rho {rho}_base',
                  ref_id=ref_id)
    else:
        dtw = DTW(ref, t, class_num=idx+1, rho=f'{rho}', starting_path=f'../data/{dataset}/rho {rho}', ref_id=ref_id)
    dtws.append(dtw)


window_size = 25
Dataset.image_creator(*dtws, window_size=window_size)

fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(len(dtws)+1, 12)

f_axi1 = fig.add_subplot(gs[0, 1:])
f_axi1.plot(timeseries, label='val')
f_axi1.plot(t.labels, label='lab')
f_axi1.axes.get_yaxis().set_visible(False)
f_axi1.axes.get_xaxis().set_visible(False)


for idx, dtw in enumerate(dtws):
    # AXIS 2
    f_axi = fig.add_subplot(gs[idx+1, 1:], sharex=f_axi1)
    img = f_axi.imshow(dtw.dtw, origin='lower', aspect='auto')
    f_axi.axes.get_yaxis().set_visible(False)
    if idx != len(dtws)-1:
        f_axi.axes.get_xaxis().set_visible(False)

    # SUB AXIS 2,1
    f_axi_1 = fig.add_subplot(gs[idx+1, 0])
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
    f_axi_1.plot(ref.lab_patterns[dtw.from_pattern_idx]['pattern'], transform=rot + base)
    if base_pattern and dtw.from_pattern_idx == 0:
        val_min = ref.lab_patterns[dtw.from_pattern_idx]['pattern'].min()
        val_max = ref.lab_patterns[dtw.from_pattern_idx]['pattern'].max()
        f_axi_1.set_xlim(-val_max*2, val_max*2)
    f_axi_1.axis('off')


fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91, 0.12, 0.01, 0.64])
fig.colorbar(img, cax=cbar_ax)

plt.show()
