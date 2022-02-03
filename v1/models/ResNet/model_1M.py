from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dense, add
from tensorflow.keras.optimizers import Adam


def get_model(x_dim, y_dim):
    n_feature_maps = 25

    input_layer = Input(x_dim)
    # BLOCK 1

    conv_x = Conv2D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv2D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv2D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv2D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = BatchNormalization(name='shortcut_1')(shortcut_y)

    output_block_1 = add([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = Conv2D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv2D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv2D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv2D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = BatchNormalization(name='shortcut_2')(shortcut_y)

    output_block_2 = add([shortcut_y, conv_z])
    output_block_2 = Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = Conv2D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv2D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv2D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = BatchNormalization(name='shortcut_3')(output_block_2)

    output_block_3 = add([shortcut_y, conv_z])
    output_block_3 = Activation('relu')(output_block_3)

    # BLOCK 4

    conv_x = Conv2D(filters=n_feature_maps * 4, kernel_size=8, padding='same')(output_block_3)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv2D(filters=n_feature_maps * 4, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv2D(filters=n_feature_maps * 4, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv2D(filters=n_feature_maps * 4, kernel_size=1, padding='same')(output_block_3)
    shortcut_y = BatchNormalization(name='shortcut_4')(shortcut_y)

    output_block_4 = add([shortcut_y, conv_z])
    output_block_4 = Activation('relu')(output_block_4)

    # FINAL

    gap_layer = GlobalAveragePooling2D()(output_block_4)

    output = Dense(y_dim, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output)

    return model


optimizer = Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)