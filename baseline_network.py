import tensorflow as tf
from tensorflow import keras
import numpy as np
import pdb
from network_utils import build_mlp
import gym

class BaselineNetwork() :
    def __init__(self, config, observation) :
        self.config = config
        self.observation = observation
        self.baseline = None
        self.lr = self.config.learning_rate

    def build(self) :
        value = build_mlp(self.observation, 1, self.config.n_layers,
                self.config.layer_size, self.config.activation)
        # value = tf.squeeze(value)
        self.baseline = keras.Model(inputs = self.observation, outputs = value)
        self.loss = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam(learning_rate = self.lr)
        self.baseline.compile(loss = self.loss, 
                              optimizer = self.optimizer)

    def calculate_advantage(self, returns, observations):
        return tf.math.subtract(returns, tf.squeeze(self.baseline(observations)))

    def update_baseline(self, returns, observations) :
        self.baseline.fit(
            x = observations,
            y = returns,
            verbose = 0,
        )