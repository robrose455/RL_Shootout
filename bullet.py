import pygame
import config


class Bullet(pygame.sprite.Sprite):

    def __init__(self, owner, player, enemy):
        super(Bullet, self).__init__()

        # Who shot the bullet
        self.owner = owner

        # Reference to the player
        self.player = player

        # Reference to the enemy
        self.enemy = enemy

        # Positional Properties
        self.radius = 5
        self.x = owner.x
        self.y = owner.y

        # Horizontal Velocity
        self.dx = 0

        if owner == player:

            # Vertical Velocity
            self.dy = -20
            self.owner = 1

        elif owner == enemy:

            # Vertical Velocity
            self.dy = 20
            self.owner = 2

    def update(self):

        # Apply Transforms
        self.y += self.dy

        # Render onto screen
        pygame.draw.circle(config.window, (255, 255, 255), (self.x, self.y), self.radius)

        # Check if out of bounds
        self.check_life_span()

        # Check for collisions
        self.check_collision()

    def check_collision(self):

        # Check if collided with enemy
        if self.owner == 1:
            if abs(self.y - self.enemy.y) <= self.enemy.radius:
                if abs(self.x - self.enemy.x) <= self.enemy.radius:
                    self.enemy.collided = True
                    self.enemy.hp -= 10
                    self.kill()

        # Check if collided with player
        if self.owner == 2:

            if abs(self.y - self.player.y) <= self.player.radius:
                if abs(self.x - self.player.x) <= self.player.radius:
                    self.player.collided = True
                    self.player.hp -= 10
                    self.kill()

    def check_life_span(self):

        # Basic Bounds Checking
        if self.y < 50:
            self.kill()

        if self.y > 500:
            self.kill()
