import pygame
import config


class Bullet(pygame.sprite.Sprite):
    def __init__(self, owner, player, enemy):

        super(Bullet, self).__init__()

        self.owner = owner

        self.player = player
        self.enemy = enemy

        self.radius = 5
        self.x = owner.x
        self.y = owner.y

        self.dx = 0
        self.dy = -20

    def update(self):

        self.y += self.dy
        pygame.draw.circle(config.window, (255, 255, 255), (self.x, self.y), self.radius)

        self.check_life_span()
        self.check_collision()

    def check_collision(self):

        if abs(self.y - self.enemy.y) <= self.enemy.radius and self.owner == self.player:
            if abs(self.x - self.enemy.x) <= self.enemy.radius:
                #print("hit")
                self.enemy.collided = True
                self.kill()

    def check_life_span(self):

        if self.y < 50:
            #print("Killed")
            self.kill()
