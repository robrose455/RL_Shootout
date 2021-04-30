import pygame
import config

ENEMY_SHOOT = pygame.USEREVENT + 1


class Enemy:

    def __init__(self):

        self.hp = 100
        self.radius = 25
        self.x = 800
        self.y = 200
        self.dx = 0
        self.dy = 0

        self.collided = False
        self.color = (214, 4, 18)
        self.countdown = 0

    def render(self):

        self.x += self.dx
        pygame.draw.circle(config.window, self.color, (self.x, self.y), self.radius)

    def update_action(self, action):

        self.dx = 0

        if action == 0:
            self.dx = -10

        if action == 1:
            self.dx = 10

        if action == 2:
            if self.countdown == 2:

                pygame.event.post(pygame.event.Event(ENEMY_SHOOT))
                self.countdown = 0

            elif self.countdown > 2:
                self.countdown = 0

            else:
                self.countdown += 1

        if self.x > config.window_width - 250:
            self.x = config.window_width - 250

        if self.x < 250:
            self.x = 250
