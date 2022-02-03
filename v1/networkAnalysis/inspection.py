import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib import cm

from tensorflow.keras import backend as K
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.callbacks import Print, GifGenerator


# Define modifier to replace a softmax function of the last layer to a linear function.
from tf_keras_vis.utils.losses import SmoothedLoss


def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear


def visualize_dense_layer(model, index):
    # Create Activation Maximization object
    activation_maximization = ActivationMaximization(model, model_modifier)

    loss = lambda x: K.mean(x[:, index-1])

    # Generate max activation with debug printing
    # Do 500 iterations and Generate an optimizing animation
    activation = activation_maximization(loss,
                                         steps=512,
                                         callbacks=[Print(interval=100),
                                                    GifGenerator('.')])
    image = activation[0].astype(np.uint8)

    f, ax = plt.subplots(figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})
    ax.imshow(image)
    plt.show()


def visualize_conv_block(model):

    layer_name = 'conv2d_3'

    # Define modifier to replace the model output to target layer's output.
    # You need to return new model when you create new model instance in model_modifier.
    def model_modifier(m):
        new_model = tf.keras.Model(inputs=m.inputs, outputs=[m.get_layer(name=layer_name).output])
        new_model.layers[-1].activation = tf.keras.activations.linear
        return new_model

    # Create Activation Maximization object
    activation_maximization = ActivationMaximization(model, model_modifier)

    num_of_filters = 16
    filter_numbers = np.random.choice(model.get_layer(name=layer_name).output.shape[-1], num_of_filters)
    cols = 4
    rows = num_of_filters // cols
    f, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4), subplot_kw={'xticks': [], 'yticks': []})

    for i, filter_number in enumerate(filter_numbers):
        # Define loss function that is sum of a filter output.
        loss = SmoothedLoss(filter_number)

        # Generate max activation
        activation = activation_maximization(loss)
        image = activation[0].astype(np.uint8)

        ax[i // cols][i % cols].imshow(image)
        ax[i // cols][i % cols].set_title(f'filter[{filter_number:03d}]')

    plt.tight_layout()
    plt.show()


def visualize_attention_sal(model, index, X, images):

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5),
                         subplot_kw={'xticks': [], 'yticks': []})
    for i in range(len(images)):
        ax[i].imshow(images[i], origin='lower')

    loss = lambda output: K.mean(output[:, index-1])

    # Create Saliency object
    saliency = Saliency(model, model_modifier)
    saliency_map = saliency(loss, X, smooth_samples=20)
    saliency_map = normalize(saliency_map)

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5), subplot_kw={'xticks': [], 'yticks': []})
    for i in range(len(saliency_map)):
        ax[i].imshow(saliency_map[i], cmap='jet', origin='lower')

    plt.show()


def visualize_attention_grad(model, index, X, images):

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5),
                         subplot_kw={'xticks': [], 'yticks': []})
    for i in range(len(images)):
        ax[i].imshow(images[i], origin='lower')

    loss = lambda output: K.mean(output[:, index-1])

    # Create Gradcam object
    gradcam = Gradcam(model, model_modifier)

    # Generate heatmap with GradCAM
    cam = gradcam(loss, X)
    cam = normalize(cam)

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5),
                         subplot_kw={'xticks': [], 'yticks': []})
    for i in range(len(cam)):
        heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
        ax[i].imshow(images[i], origin='lower')
        ax[i].imshow(heatmap,  origin='lower', cmap='jet', alpha=0.5)

    plt.show()

