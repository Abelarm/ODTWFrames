from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow. keras.optimizers import Adam


def get_model(x_dim, y_dim):

    model = Sequential([
        GRU(input_shape=x_dim, units=32, dropout=0.2, recurrent_dropout=0.2,
            return_sequences=True),
        GRU(units=64, dropout=0.2, recurrent_dropout=0.35),
        Dense(128, activation='relu'),
        Dense(y_dim, activation='softmax')
    ])

    return model


optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
