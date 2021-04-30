import gym
from gym import spaces
import config
import tensorflow as tf
import neural_network
import tensorflow_probability as tfp
import numpy as np
import pygame

RESET = pygame.USEREVENT + 2


class Environment(gym.Env):

    def __init__(self, host, player, bullets):

        self.n_of_actions = 3

        # Reference to player and enemy
        self.host = host
        self.player = player
        self.bullets = bullets

        self.step_count = 0
        self.enemy_hp_bar = (0, 0, 0)
        self.player_hp_bar = (0, 0, 0)

        # Reference to the network
        self.nn = neural_network.NeuralNetwork(n_actions=self.n_of_actions)

        # Define the observation and action space
        self.observation_space = spaces.Box(np.array([[-1, -1]]), np.array([[1, 1]]), shape=(1, 2))
        self.action_space = spaces.Discrete(self.n_of_actions)

        # Action Probabilities
        self.prob_left = 0
        self.prob_right = 0
        self.prob_shoot = 0

        # State of Environment ( X pos )
        self.observation = None

        # Action of the agent to be processed
        self.action = 0

        # Reward of the agent
        self.reward = 0

        # Discount factor
        self.gamma = 0.5

        # Terminal Flag
        self.done = False

        self.reward_mode = 0

        self.info = None

    # Reset Environment to Default Values
    def reset(self):

        self.host.x = 800
        self.player.x = 800

        if self.player.hp >= 0:
            self.host.hp = 100
            self.player.hp = 100

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

        # print(probs)

        # Extract Probs to Display to UI
        prob_numpy = probs.numpy()
        self.prob_left = round(prob_numpy[0][0][0], 4)
        self.prob_right = round(prob_numpy[0][0][1], 4)
        self.prob_shoot = round(prob_numpy[0][0][2], 4)

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

        # Adjust Weights and Bias based on old state. new state, and reward given
        with tf.GradientTape() as tape:
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

    def calculate_reward(self, mode):

        self.reward += -1

        # Different Demo Reward Modes
        if mode == 1:

            if self.host.dx < 0:
                self.reward += 2

            if self.host.dx > 0:
                self.reward += -2

            print(self.reward)

        if mode == 2:

            if self.host.dx < 0:
                self.reward += -2

            if self.host.dx > 0:
                self.reward += 2

        if mode == 3:

            center_discount = abs(self.host.x - self.player.x) / self.player.x

            if self.host.dx < 0:
                self.reward += +2 * (1 - center_discount)

            if self.host.dx > 0:
                self.reward += +2 * (1 - center_discount)

            if self.host.x == self.player.x:
                self.reward += 20

        if mode == 4:

            self.reward = 0

            if self.host.collided:
                self.reward = -10

            if self.player.collided:
                self.reward += 5

    def step(self, action):

        self.reward = 0

        # perform one step in the game logic
        self.enemy_hp_bar = (0, 0, 0)
        self.player_hp_bar = (0, 0, 0)

        self.step_count += 1

        for b in self.bullets:
            b.update()

        self.host.update_action(action)

        # Render Updated Positions
        self.render()
        self.host.render()
        self.player.render()

        # Reward Structure
        self.calculate_reward(self.reward_mode)

        if self.host.collided:
            self.host.collided = False
            self.enemy_hp_bar = (255, 0, 0)

        if self.player.collided:
            self.player.collided = False
            self.player_hp_bar = (255, 0, 0)

        if self.host.hp <= 0:
            self.reset()

        if self.player.hp <= 0:
            pygame.event.post(pygame.event.Event(RESET))

        if self.player.hp <= 0:
            self.done = True

        # X pos of agent
        x_enemy_norm = np.interp(self.host.x, [0, 1000], [0, 1])

        # X pos of player
        x_player_norm = np.interp(self.player.x, [0, 1000], [0, 1])

        my_tensor = tf.constant([[x_enemy_norm, x_player_norm]])
        my_variable = tf.Variable(my_tensor, dtype=np.float64)
        observation = my_variable

        return observation, self.reward, self.done, self.info

    def render(self):
        pass
