from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow. keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def get_model(x_dim, y_dim):

    model = Sequential([
        LSTM(input_shape=x_dim, units=8, dropout=0.2, recurrent_dropout=0.2,
             return_sequences=True),
        LSTM(units=16, dropout=0.2, recurrent_dropout=0.2,
             return_sequences=True),
        LSTM(units=32, dropout=0.2, recurrent_dropout=0.2,
             return_sequences=True),
        BatchNormalization(momentum=0.999, epsilon=0.01),
        LSTM(units=64, dropout=0.2, recurrent_dropout=0.35,
             return_sequences=True),
        LSTM(units=128, dropout=0.2, recurrent_dropout=0.35),
        BatchNormalization(momentum=0.999, epsilon=0.01),
        Dense(128, activation='relu',
              kernel_initializer="he_normal",
              kernel_regularizer=l2(0.01)),
        Dense(y_dim, activation='softmax')
    ])

    return model


optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
