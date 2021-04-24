import pygame
import config


class Enemy:

    def __init__(self):
        self.radius = 25

        self.x = 500
        self.y = 400

        self.dx = 0
        self.dy = 0

        self.action = None

    def render(self):
        color2 = (255, 255, 255)
        # make draw call

        pygame.draw.circle(config.window, color2, (self.x, self.y), self.radius)  # draw a cir

    def update(self, action):

        self.dx = 0
        if action == 0:
            self.dx = -10

        if action == 1:
            self.dx = 10

        if action == -1:
            print("Setting Speed to 0")
            self.dx = 0

        self.x += self.dx

        if self.x > config.window_width:
            self.x = config.window_width
            self.dx = -self.dx

        if self.x < 0:
            self.x = 0
            self.dx = -self.dx

