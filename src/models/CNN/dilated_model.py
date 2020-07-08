from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.layers import MaxPooling2D


def get_model(x_dim, y_dim):
    # Define the input

    input_layer = Input(shape=x_dim)

    conv_1 = Conv2D(32, 3, padding='same',
                    dilation_rate=1,
                    activation='relu',
                    kernel_initializer="he_normal",
                    input_shape=x_dim)(input_layer)

    conv_2 = Conv2D(64, 3, padding='same',
                    dilation_rate=2,
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(0.1))(conv_1)
    batch_1 = BatchNormalization(momentum=0.999, epsilon=0.01)(conv_2)

    conv_3 = Conv2D(128, 3, padding='same',
                    activation='relu',
                    dilation_rate=4,
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(0.1))(batch_1)

    conv_4 = Conv2D(256, 3, padding='same',
                    kernel_initializer="he_normal")(conv_3)
    batch_2 = BatchNormalization(momentum=0.999, epsilon=0.01)(conv_4)
    act_2 = Activation('relu')(batch_2)

    glob = MaxPooling2D()(act_2)

    flat = Flatten()(glob)
    dense_1 = Dense(256, activation='relu',
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(0.01))(flat)
    dense_2 = Dense(128, activation='relu',
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(0.01))(dense_1)

    output = Dense(y_dim, activation='softmax')(dense_2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    return model


optimizer = Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
