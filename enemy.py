import pygame
import config


class Enemy:

    def __init__(self):

        self.radius = 25

        self.x = 500
        self.y = 100

        self.dx = 0
        self.dy = 0

        self.action = None
        self.collided = False

    def render(self):

        color2 = (0, 0, 255)
        # make draw call

        pygame.draw.circle(config.window, color2, (self.x, self.y), self.radius)  # draw a cir

        self.x += self.dx

    def update_action(self, action):

        self.dx = 0

        if action == 0:
            self.dx = -10

        if action == 1:
            self.dx = 10

        if action == -1:
            self.dx = 0

        if self.x > config.window_width - 25:
            self.x = config.window_width - 25
            self.dx = -self.dx

        if self.x < 25:
            self.x = 25
            self.dx = -self.dx

