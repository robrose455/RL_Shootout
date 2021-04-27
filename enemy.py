import pygame
import config

ENEMY_SHOOT = pygame.USEREVENT + 1


class Enemy:

    def __init__(self):

        self.hp = 100
        self.radius = 25
        self.x = 500
        self.y = 100
        self.dx = 0
        self.dy = 0

        self.collided = False
        self.color = (0, 0, 255)
        self.unknown_glitch = 0

    def render(self):

        self.x += self.dx
        pygame.draw.circle(config.window, self.color, (self.x, self.y), self.radius)

    def update_action(self, action):

        # idk what the hell is happening here
        # ignores the first call
        if self.unknown_glitch == 1:


            # Weird Data
            # print(action)
            self.dx = 0

            if action[0] == 0:
                self.dx = -10

            if action[0] == 1:
                self.dx = 10

            if action[0] == 2:
                pygame.event.post(pygame.event.Event(ENEMY_SHOOT))

            if self.x > config.window_width - 25:
                self.x = config.window_width - 25
                self.dx = -self.dx

            if self.x < 25:
                self.x = 25
                self.dx = -self.dx

        else:

            self.unknown_glitch += 1
