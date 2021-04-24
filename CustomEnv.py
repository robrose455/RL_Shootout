import gym
from gym import spaces
import pygame
import config
import tensorflow as tf
import neural_network
import tensorflow_probability as tfp
import numpy as np


class CustomEnv(gym.Env):

    def __init__(self, host):

        self.host = host

        self.nn = neural_network.NeuralNetwork(2)

        high = np.array([1], dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self.observation = None
        self.reward = 0
        self.info = None
        self.done = False
        self.action = 0

        self.score = 0

        self.gamma = 0.99

    def reset(self):
        pass

    def choose_action(self, observation):

        state = observation
        state = tf.expand_dims(state, 0)
        _, probs = self.nn(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)

        action = action_probabilities.sample()
        self.action = action

        return action.numpy()[0]

    def learn(self, state, reward, state_, done):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:

            state_value, probs = self.nn(state)
            state_value_, _ = self.nn(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            print(probs)

            action_probs = tfp.distributions.Categorical(probs=probs)

            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.nn.trainable_variables)
        self.nn.optimizer.apply_gradients(zip(gradient, self.nn.trainable_variables))

    def step(self, action):
        # perform one step in the game logic

        self.host.update_action(action)

        if self.host.collided:

            self.reward = -50
            self.host.collided = False

        elif self.host.x > config.window_width - 50:
            self.reward = -100

        elif self.host.x < 50:
            self.reward = -100

        else:

            self.reward = 1

        return self.observation, self.reward, self.done, self.info

    def render(self):

        color = (0, 0, 0)
        config.window. fill(color)
