import gym
import tensorflow as tf
from tensorflow import keras
import os
from general import Progbar, export_plot
from network_utils import build_mlp
import numpy as np

class Normal(keras.layers.Layer):
    def __init__(self,) :
        super().__init__()

    def build(self, action_dim):
        self.action_dim = action_dim[-1]
        self.log_std = self.add_weight(shape = [self.action_dim],
                                       initializer = 'random_normal',
                                       trainable = True)
        
    def call(self, inputs) :
        r = keras.backend.random_normal([self.action_dim])
        x = tf.math.multiply(r, tf.math.exp(self.log_std))
        x = tf.math.add(x, inputs)
        return x

observation = keras.Input(
    dtype = tf.float32,
    shape = (4,),
)
action = keras.layers.Dense(4)(observation)
n_layer = Normal()
sampled_action = n_layer(action)
model = keras.Model(inputs = observation, outputs = sampled_action, name = 'sample_action')
model.summary()
print('n_layer log_std :{}'.format(n_layer.log_std))
for _ in range(10) :
    print(model(tf.ones([4,4])))
