from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def get_model(x_dim, y_dim):
    # Define the input
    input_layer = Input(shape=x_dim)

    conv_1 = Conv2D(32, 3, padding='same', activation='relu',
                    kernel_initializer="he_normal",
                    input_shape=x_dim)(input_layer)
    max_1 = MaxPooling2D()(conv_1)

    conv_2 = Conv2D(64, 3, padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(0.1))(max_1)
    batch_1 = BatchNormalization(momentum=0.999, epsilon=0.01)(conv_2)
    act_1 = Activation('relu')(batch_1)
    max_2 = MaxPooling2D()(act_1)

    conv_3 = Conv2D(128, 3, padding='same',
                    activation='relu',
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(0.1))(max_2)

    conv_4 = Conv2D(256, 3, padding='same',
                    kernel_initializer="he_normal")(conv_3)
    batch_2 = BatchNormalization(momentum=0.999, epsilon=0.01)(conv_4)
    act_2 = Activation('relu')(batch_2)

    print(act_2.shape)
    if act_2.shape[2] == 1:
        max_3 = act_2
    else:
        max_3 = MaxPooling2D()(act_2)

    flat = Flatten()(max_3)
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
