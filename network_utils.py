import tensorflow as tf
from tensorflow import keras

def build_mlp(
    mlp_input,
    output_size,
    n_layers,
    size,
    output_activation = None):

    x = mlp_input
    for _ in range(n_layers) :
        x = keras.layers.Dense(units = size, activation = 'relu')(x)
    return keras.layers.Dense(units = output_size, activation = output_activation)(x)