from os.path import join

import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tf_explain.core import ExtractActivations


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
        f_axi.imshow(predictions[:, :, c], cmap='plasma', origin='lower', aspect='auto', interpolation='bilinear')
        f_axi.axes.get_yaxis().set_visible(False)
        f_axi.axes.get_xaxis().set_visible(False)
        f_axi.set_title(f'#f: {c}')

    plt.savefig(join(save_dir, f'Filters of layer:{layer_name} for class {class_num}.pdf'), bbox_inches='tight')
    plt.close(fig)


def visualize_gradients(model, generator, class_num, save_dir):
    for x, y in generator:

        if y.argmax() == class_num:
            selected_x = x
            break

    expected_output = y

    with tf.GradientTape() as tape:
        inputs = tf.cast(selected_x, tf.float32)
        tape.watch(inputs)
        predictions = model(inputs)
        loss = tf.keras.losses.categorical_crossentropy(
            expected_output, predictions
        )

    grad = tape.gradient(loss, inputs)
    input_grad = tf.multiply(inputs, grad)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    row = -1

    for c in range(input_grad.shape[-1]):

        if c % 4 == 0:
            row += 1
        f_axi = fig.add_subplot(gs[row, c % 4])

        img = f_axi.imshow(input_grad[0, :, :, c], cmap='plasma', origin='lower', aspect='auto')
        f_axi.axes.get_yaxis().set_visible(False)
        f_axi.axes.get_xaxis().set_visible(False)
        f_axi.set_title(f'pattern: {c}')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.99, 0.05, 0.01, 0.81])
    fig.colorbar(img, cmap='plasma', cax=cbar_ax)
    plt.savefig(join(save_dir, f'Channel gradients for image of class {class_num}.pdf'), bbox_inches='tight')
    plt.close(fig)


def make_gradcam_heatmap(model, generator, class_num, save_dir, last_conv_layer_name='conv2d_3'):

    for x, y in generator:

        if y.argmax() == class_num:
            selected_x = x
            break

    if selected_x.shape[2] == 5:
        classifier_layer_names = ['batch_normalization_1', 'activation_1', 'flatten', 'dense',
                                  'dense_1', 'dense_2']
    elif selected_x.shape[2] > 5:
        classifier_layer_names = ['batch_normalization_1', 'activation_1', 'max_pooling2d_2',
                                  'flatten', 'dense', 'dense_1', 'dense_2']

    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(selected_x)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    plt.imshow(heatmap, aspect='auto', origin='lower')
    plt.savefig(join(save_dir, f'GradCAM for class {class_num}.pdf'), bbox_inches='tight')