import re
from os.path import join

import matplotlib.pyplot as plt

from dataset.files import TimeSeries
from networkAnalysis.explain import _get_gradcam_heatmap, _get_gradients
from utils.functions import get_id_interval

core_path = '../../..'


def explain_sample(dataset_name, relevant_sample_name, selected_x, selected_y, model, class_num, save_dir):

    ts_id, interval = get_id_interval(relevant_sample_name)

    filename = f'{core_path}/data/{dataset_name}/' \
                f'STREAM_length-100_noise-5_warp-10_shift-10_outliers-0_cycles-per-label-10_set' \
                f'-test_id-{ts_id}.npy'

    if 'gunpoint' in dataset_name:
        filename = f'{core_path}/data/{dataset_name}/' \
                    f'STREAM_cycles-per-label-20_set-test_id-{ts_id}.npy'

    t = TimeSeries(filename)
    timeseries = t.timeseries[interval[0]: interval[1]]

    gradCAM_heatmap = _get_gradcam_heatmap(model, selected_x)
    print(gradCAM_heatmap.shape)
    gradients = _get_gradients(model, selected_x, selected_y, mult_input=False)[0]
    print(gradients.shape)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(4, gradients.shape[-1])

    f_axi_0 = fig.add_subplot(gs[0, :])
    f_axi_0.plot(timeseries)
    f_axi_0.axes.get_yaxis().set_visible(False)
    f_axi_0.axes.get_xaxis().set_visible(False)
    f_axi_0.axis(xmin=0, xmax=len(timeseries)-1)

    for c in range(selected_x.shape[-1]):
        f_axi_1 = fig.add_subplot(gs[1, c])
        img = f_axi_1.imshow(selected_x[0, :, :, c], aspect='auto', origin='lower', cmap='plasma')
        f_axi_1.axes.get_yaxis().set_visible(False)
        f_axi_1.axes.get_xaxis().set_visible(False)

    plt.colorbar(img, ax=f_axi_1, cmap='plasma')

    f_axi_1 = fig.add_subplot(gs[2, :])
    img = f_axi_1.imshow(gradCAM_heatmap, aspect='auto', origin='lower')
    f_axi_1.axes.get_yaxis().set_visible(False)
    f_axi_1.axes.get_xaxis().set_visible(False)

    for c in range(gradients.shape[-1]):

        f_axi_1 = fig.add_subplot(gs[3, c])
        img = f_axi_1.imshow(gradients[:, :, c], aspect='auto', origin='lower', cmap='plasma')
        f_axi_1.axes.get_yaxis().set_visible(False)
        f_axi_1.axes.get_xaxis().set_visible(False)

    plt.colorbar(img, ax=f_axi_1, cmap='plasma')
    plt.savefig(join(save_dir, f'Relevant example for class {class_num}.pdf'), bbox_inches='tight')
    plt.close(fig)