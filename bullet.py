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

        if owner == player:

            self.dy = -40
            self.owner = 1

        elif owner == enemy:

            self.dy = 40
            self.owner = 2

    def update(self):

        self.y += self.dy
        pygame.draw.circle(config.window, (255, 255, 255), (self.x, self.y), self.radius)

        self.check_life_span()
        self.check_collision()

    def check_collision(self):

        if self.owner == 1:
            if abs(self.y - self.enemy.y) <= self.enemy.radius:
                if abs(self.x - self.enemy.x) <= self.enemy.radius:
                    self.enemy.collided = True
                    self.enemy.hp -= 10
                    self.kill()

        if self.owner == 2:

            if abs(self.y - self.player.y) <= self.player.radius:
                print("Dub")
                if abs(self.x - self.player.x) <= self.player.radius:
                    self.player.collided = True
                    self.player.hp -= 10
                    self.kill()


    def check_life_span(self):

        if self.y < 50:
            self.kill()

        if self.y > 500:
            self.kill()
