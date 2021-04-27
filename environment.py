import gym
from gym import spaces
import pygame
import config
import tensorflow as tf
import neural_network
import tensorflow_probability as tfp
import numpy as np


class Environment(gym.Env):

    def __init__(self, host, player):

        self.n_of_actions = 3

        # Reference to player and enemy
        self.host = host
        self.player = player

        # Reference to the network
        self.nn = neural_network.NeuralNetwork(n_actions=self.n_of_actions)

        # Define the observation and action space
        high = np.array([1], dtype=np.float32)
        self.observation_space = spaces.Box(np.array([[-1], [-1]]), np.array([[1], [1]]), shape=(2, 1))
        self.action_space = spaces.Discrete(self.n_of_actions)

        # State of Environment ( X pos )
        self.observation = None

        # Action of the agent to be processed
        self.action = 0

        # Reward of the agent
        self.reward = 0

        # Discount factor
        self.gamma = 0.99

        # Terminal Flag
        self.done = False

        self.info = None

    # Reset Environment to Default Values
    def reset(self):

        self.host.x = 500
        self.player.x = 500

        if self.player.hp >= 0:
            self.host.hp = 100

        # Increment Level
        config.level += 1

    # Choose Action Based On Given State
    def choose_action(self, observation):

        self.observation = observation
        # Convert Observation Data to State
        state = observation

        # Fit for model
        state = tf.expand_dims(state, 0)

        # Input: State of Environment
        # Output: Raw Action Probabilities
        _, probs = self.nn(state)

        # Distrubution of Action Probs
        action_probabilities = tfp.distributions.Categorical(probs=probs)

        # Pick an Action
        action = action_probabilities.sample()
        self.action = action

        return action.numpy()[0]

    # Apply data to adjust network
    def learn(self, state, reward, state_, done):

        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.nn(state)
            state_value_, _ = self.nn(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

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

        #print("-------")
        #print(self.host.x)
        # Render Updated Positions
        self.render()
        self.host.render()
        self.player.render()

        if self.host.collided:

            self.reward = -500
            self.host.collided = False

        elif self.host.x > config.window_width - 50:
            self.reward = -100

        elif self.host.x < 50:
            self.reward = -100

        if self.player.collided:
            self.reward += 5000
            self.player.collided = False

        else:

            self.reward = -50

        if self.host.hp <= 0 or self.player.hp <= 0:
            self.reset()

        if self.player.hp <= 0:
            print("True")
            self.done = True

        # X pos of agent
        x_enemy_norm = np.interp(self.host.x, [0, 1000], [0, 1])

        # X pos of player
        x_player_norm = np.interp(self.player.x, [0, 1000], [0, 1])

        my_tensor = tf.constant([[x_enemy_norm], [x_player_norm]])
        my_variable = tf.Variable(my_tensor, dtype=np.float64)
        observation = my_variable

        #print(self.host.x)

        return observation, self.reward, self.done, self.info

    def render(self):

        color = (0, 0, 0)
        config.window.fill(color)
