import gym
import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
from policy_graident import PG
from config import get_config
import pybullet
import pybullet_envs

pybullet.connect(pybullet.DIRECT)
env_name = 'pendulum'
use_baseline = True
r_seed = 35

tf.random.set_seed(r_seed)
np.random.seed(r_seed)
random.seed(r_seed)

config = get_config(env_name, use_baseline, r_seed)
env = gym.make(config.env_name)
model = PG(env, config, r_seed)
model.run()