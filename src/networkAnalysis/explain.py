from os.path import join

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_explain.core import ExtractActivations, GradientsInputs
from tf_explain.utils.image import transform_to_normalized_grayscale


def visualize_activation(model, generator, class_num, save_dir, layer_name='conv2d'):

    for x, y in generator:

        if y.argmax() == class_num:
            selected_x = x
            break

    explainer = ExtractActivations(batch_size=1)

    activation_model = explainer.generate_activations_graph(model, layer_name)
    predictions = activation_model.predict(selected_x, batch_size=1)
    if type(predictions) is list:
        predictions = predictions[-1]
    predictions = predictions[0]

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(4, 8)

    row = -1
    for c in range(predictions.shape[-1]):
        if c == 32:
            break
        if c % 8 == 0:
            row += 1
        f_axi = fig.add_subplot(gs[row, c % 8])
        f_axi.imshow(predictions[:, :, c], cmap='plasma', origin='lower', aspect='auto')
        f_axi.axes.get_yaxis().set_visible(False)
        f_axi.axes.get_xaxis().set_visible(False)
        f_axi.set_title(f'#f: {c}')

    plt.savefig(join(save_dir, f'Filters of layer:{layer_name} for class {class_num}.pdf'), bbox_inches='tight')
    plt.close(fig)


def visualize_gradients(model, generator, class_num, save_dir):

    i = 0
    batch_x = []
    for x, y in generator:

        if y.argmax() == class_num:
            batch_x.append(x[0])
            i += 1

        if i == 32:
            break

    batch_x = np.stack(batch_x, axis=0)
    explainer = GradientsInputs()

    gradients = explainer.compute_gradients(batch_x, model, class_num)
    grayscale_gradients = transform_to_normalized_grayscale(
        tf.abs(gradients)
    ).numpy()

    grayscale_gradients = np.mean(grayscale_gradients, axis=0)

    fig = plt.figure(constrained_layout=True)
    plt.imshow(grayscale_gradients, origin='lower', aspect='auto', cmap='Greys', interpolation='bilinear')
    plt.title(f'Input gradients for image of class {class_num}')
    plt.savefig(join(save_dir, f'Input gradients for image of class {class_num}.pdf'), bbox_inches='tight')
    plt.close(fig)
