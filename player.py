import config
import pygame


class Player:

    def __init__(self):

        self.radius = 25

        self.hp = 100

        self.x = 800
        self.y = 600

        self.dx = 0
        self.dy = 0

        self.color = (37, 5, 153)

        self.collided = False

        self.action = None

    def render(self):

        pygame.draw.circle(config.window, self.color, (self.x, self.y), self.radius)  # draw a circle

    def update(self, action):

        self.dx = 0

        if self.collided:

            self.hp -= 10

        if action == 0:
            self.dx = -10

        if action == 1:
            self.dx = 10

        if action == -1:
            self.dx = 0

        self.x += self.dx

        if self.x > config.window_width - 250:
            self.x = config.window_width - 250
            self.dx = -self.dx

        if self.x < 250:
            self.x = 250
            self.dx = -self.dx
