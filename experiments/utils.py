from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Input


def build_encoder(input_shape, units, n_components, activation='relu'):
    encoder = Sequential([
        Input(shape=input_shape),
        Dense(units, activation=activation),
        Dense(units//2, activation=activation),
        Dense(units//4, activation=activation),
        BatchNormalization(),
        Dense(n_components, use_bias=False, activation='linear')
    ], name='encoder')

    return encoder


def build_decoder(output_shape, units, n_components, activation='relu'):
    decoder = Sequential([
        Input(shape=(n_components,)),
        Dense(units//4, activation=activation),
        Dense(units//2, activation=activation),
        Dense(units, activation=activation),
        Dense(*output_shape, activation='linear')
    ], name='decoder')

    return decoder