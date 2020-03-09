import pybullet
import pybullet_envs
import gym
import time

pybullet.connect(pybullet.DIRECT)
env = gym.make('CartPoleBulletEnv-v1')
env = gym.wrappers.Monitor(env, "recording", force = True)
env.render()
env.reset()
for _ in range(100) :
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.1)

env.close()