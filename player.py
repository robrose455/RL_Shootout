import config
import pygame


class Player:

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

        pygame.draw.circle(config.window, color2, (self.x, self.y), self.radius)  # draw a circle

    def update(self, action):

        self.dx = 0
        if action == 0:
            self.dx = -10

        if action == 1:
            self.dx = 10

        if action == -1:
            self.dx = 0

        self.x += self.dx

        if self.x > config.window_width - 50:
            self.x = config.window_width - 50
            self.dx = -self.dx

        if self.x < 50:
            self.x = 50
            self.dx = -self.dx
