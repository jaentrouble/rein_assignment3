import gym
import tensorflow as tf
from tensorflow import keras
import os
from general import export_plot
from network_utils import build_mlp
import numpy as np
import tensorflow_probability as tfp
from baseline_network import BaselineNetwork
import sys

class Normal_action_sample(keras.layers.Layer):
    def __init__(self,) :
        super().__init__()

    def build(self, action_dim):
        self.action_dim = action_dim[-1]
        self.log_std = self.add_weight(shape = [self.action_dim],
                                       initializer = 'random_uniform',
                                       trainable = True)
        
    def call(self, inputs) :
        r = keras.backend.random_normal([self.action_dim])
        x = tf.math.multiply(r, tf.math.exp(self.log_std))
        x = tf.math.add(x, inputs)
        return x

class PG() :
    def __init__(self,env : gym.Env, config, r_seed) :
        if not os.path.exists(config.output_path) :
            os.makedirs(config.output_path)
        self.config = config
        self.r_seed = r_seed
        self.env = env
        self.env.seed(self.r_seed)

        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        self.lr = self.config.learning_rate
        self.build()
        self.set_summary()
        self.set_optimizer()

    def build(self) :
        self.observation = keras.Input(
            dtype = tf.float32,
            shape = (self.observation_dim,),
        )
        self.action = build_mlp(self.observation, self.action_dim, self.config.n_layers,
                        self.config.layer_size, self.config.activation)
        self.action_logit = keras.Model(inputs = self.observation, outputs = self.action)

        if self.discrete :
            sampled_action = tf.squeeze(tf.random.categorical(self.action,1))
        else :
            self.normal_layer = Normal_action_sample()
            sampled_action = self.normal_layer(self.action)

        self.sample_action = keras.Model(inputs = self.observation, outputs = sampled_action, name='sample_action')
        self.sample_action.summary()
        if self.config.use_baseline :
            self.baseline_network = BaselineNetwork(self.config, self.observation)
            self.baseline_network.build()
            
    def set_optimizer(self) :
        self.optimizer = keras.optimizers.Adam()

    def loss_func(self, observations, actions, advantages) :
        if self.discrete :
            self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(actions, self.action_logit(observations))
        else :
            self.logprob = tfp.distributions.MultivariateNormalDiag(
                loc = self.action_logit(observations),
                scale_diag= tf.math.exp(self.normal_layer.log_std),
            ).log_prob(actions)
        advantages = tf.cast(advantages, tf.float32)
        return -tf.math.multiply(self.logprob, advantages)

    def train_step(self, observations, actions, advantages) :
        with tf.GradientTape() as tape :
            self.loss = self.loss_func(observations, actions, advantages)
        gradients = tape.gradient(self.loss, self.sample_action.trainable_variables)
        self.optimizer.learning_rate = self.lr
        self.optimizer.apply_gradients(zip(gradients, self.sample_action.trainable_variables))

    def set_summary(self) :
        self.file_writer = tf.summary.create_file_writer(self.config.output_path)
        self.file_writer.set_as_default()
    
    def get_returns(self, paths) :
        all_returns = []
        for path in paths :
            rewards = path["reward"]
            gammas = self.config.gamma**np.arange(len(rewards))
            returns = np.flip(np.cumsum(np.flip(rewards * gammas)))
            rev_gammas = self.config.gamma**(-np.arange(len(rewards)))
            returns = rev_gammas * returns
            all_returns.append(returns)
        returns = np.concatenate(all_returns)

        return returns

    def normalize_advantage(self, advantages) :
        adv = (advantages - tf.math.reduce_mean(advantages))/tfp.stats.stddev(advantages)
        return adv

    def calculate_advantage(self, returns, observations) :
        if self.config.use_baseline :
            advantages = self.baseline_network.calculate_advantage(returns, observations)
        else :
            advantages = returns
        
        if self.config.normalize_advantage :
            advantages = self.normalize_advantage(advantages)
        
        return advantages

    def add_sumary(self, t) :
        tf.summary.scalar('Avg Reward', self.avg_reward, step= t)
        tf.summary.scalar('Max Reward', self.max_reward, step= t)
        tf.summary.scalar('Std Reward', self.std_reward, step= t)
        tf.summary.scalar('Eval Reward', self.eval_reward, step= t)

    def init_averages(self) :
        self.avg_reward = 0
        self.max_reward = 0
        self.std_reward = 0
        self.eval_reward = 0

    def update_averages(self, rewards, scores_eval) :
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards)/ len(rewards))

        if len(scores_eval) > 0 :
            self.eval_reward = scores_eval[-1]

    def sample_path(self, env, num_episodes = None) :
        episode = 0
        episode_rewards = []
        paths = []
        t = 0
        while (num_episodes or t < self.config.batch_size) :
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len) :
                states.append(state)
                action = self.sample_action(states[-1][None])
                if self.discrete :
                    action = int(action)
                else :
                    action = action[0]
                state, reward, done, info = env.step(action)
                # env.render()
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if (done or step == self.config.max_ep_len-1) :
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size :
                    break

            path = {
                'observation' : np.array(states),
                'reward' : np.array(rewards),
                'action' : np.array(actions)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes :
                break
        
        return paths, episode_rewards

    def train(self) :
        last_eval = 0
        last_record = 0
        scores_eval = []

        self.init_averages()

        for t in range(self.config.num_batches) :
            paths, total_rewards = self.sample_path(self.env)
            scores_eval = scores_eval + total_rewards
            observations = np.concatenate([path['observation'] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            returns = self.get_returns(paths)

            advantages = self.calculate_advantage(returns, observations)

            if self.config.use_baseline :
                self.baseline_network.update_baseline(returns, observations)

            self.train_step(observations, actions, advantages)

            if (t % self.config.summary_freq == 0) :
                self.update_averages(total_rewards, scores_eval)
                self.add_sumary(t)

            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards)/len(total_rewards))
            sys.stdout.write('\r')
            sys.stdout.flush()
            msg = "Average reward: {0:04.2f} +/- {1:04.2f} step:{2}/{3}         ".format(avg_reward, sigma_reward,
                  t, self.config.num_batches)
            print(msg, end='')

            if  self.config.record and not ((t+1)% self.config.record_freq):
                sys.stdout.write('\n')
                sys.stdout.flush()
                print('Recording')
                self.record()

        sys.stdout.write('\n')
        sys.stdout.flush()
        print('Training done.')
        print(self.normal_layer.log_std.numpy())
        export_plot(scores_eval, 'Score', self.config.env_name, self.config.plot_output)
        
    def evaluate(self, env= None, num_episodes=1):
        if env==None : env = self.env
        self.sample_path(env, num_episodes)

    def record(self) :
        env = gym.make(self.config.env_name)
        env.seed(self.r_seed)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable= lambda x: True, resume=True)
        self.evaluate(env, 1)

    def run(self) :
        if self.config.record :
            self.record()
        self.train()
        if self.config.record :
            self.record()