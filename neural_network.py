import os
from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class NeuralNetwork(keras.Model, ABC):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')

        # Critic
        self.v = Dense(1, activation=None)

        # Actor
        self.pi = Dense(3, activation='softmax')

    def __call__(self, state):

        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi
