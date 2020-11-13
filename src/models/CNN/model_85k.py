from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dense, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def get_model(x_dim, y_dim):
    # Define the input
    input_layer = Input(shape=x_dim)

    conv_2 = Conv2D(64, 3, padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(0.1))(input_layer)
    batch_1 = BatchNormalization(momentum=0.999, epsilon=0.01)(conv_2)
    act_1 = Activation('relu')(batch_1)

    if act_1.shape[2] == 1:
        max_2 = act_1
    else:
        max_2 = MaxPooling2D()(act_1)

    conv_4 = Conv2D(128, 3, padding='same',
                    kernel_initializer="he_normal")(max_2)
    batch_2 = BatchNormalization(momentum=0.999, epsilon=0.01)(conv_4)
    act_2 = Activation('relu')(batch_2)

    print(act_2.shape)
    if act_2.shape[2] == 1:
        max_3 = act_2
    else:
        max_3 = MaxPooling2D()(act_2)

    flat = Flatten()(max_3)

    output = Dense(y_dim, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    return model


optimizer = Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
