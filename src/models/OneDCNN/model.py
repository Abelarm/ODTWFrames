from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dropout, BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def get_model(x_dim, y_dim):
    model = Sequential([
        Conv1D(16, 3, padding='valid',
               activation='relu',
               input_shape=x_dim),
        MaxPooling1D(),
        Conv1D(32, 3, padding='same', activation='relu'),
        AveragePooling1D(padding='same'),
        Dropout(0.35),
        Conv1D(64, 3, padding='same'),
        BatchNormalization(momentum=0.999, epsilon=0.01),
        Activation('relu'),
        Dropout(0.4),
        Conv1D(128, 3, padding='same', activation='relu',
               kernel_regularizer=l2(0.001)),
        Flatten(),
        Dense(128, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dense(256, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dense(y_dim, activation='softmax')
    ])

    return model


optimizer = Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
