import CustomEnv
import pygame
import config
import player
import bullet
import enemy
import numpy as np
import tensorflow as tf

from pygame.locals import (

    K_LEFT,
    K_RIGHT,
    KEYDOWN,
    QUIT,
    K_SPACE,
    K_r

)

config.window_height = 500
config.window_width = 1000
config.level = 1

if __name__ == '__main__':

    ENEMY_SHOOT = pygame.USEREVENT + 1

    pygame.init()

    clock = pygame.time.Clock()
    config.window = pygame.display.set_mode((config.window_width, config.window_height))

    frame_buffer = 0
    p = player.Player()
    e = enemy.Enemy()

    bullets = pygame.sprite.Group()

    env = CustomEnv.CustomEnv(e, p)

    score = 0

    run = True
    p_action = -1
    ai_action = 0

    font = pygame.font.Font('freesansbold.ttf', 15)
    player_hp = font.render(str(p.hp), False, (0, 0, 0))
    player_hp_rect = player_hp.get_rect()
    player_hp_rect.center = (50, 450)

    enemy_hp = font.render(str(e.hp), False, (0, 0, 0))
    enemy_hp_rect = enemy_hp.get_rect()
    enemy_hp_rect.center = (50, 50)

    level_text = font.render(str(config.level), False, (0, 0, 0))
    level_text_rect = level_text.get_rect()
    level_text_rect.center = (50, 250)

    while run:

        # Event Manager
        for event in pygame.event.get():

            if event.type == KEYDOWN:

                if event.key == K_LEFT:
                    p_action = 0

                if event.key == K_RIGHT:
                    p_action = 1

                if event.key == K_SPACE:
                    b = bullet.Bullet(p, p, e)
                    bullets.add(b)

                if event.key == K_r:
                    env.reset()

            if event.type == pygame.KEYUP:

                if event.key == K_LEFT:
                    p_action = -1

                if event.key == K_RIGHT:
                    p_action = -1

            if event.type == ENEMY_SHOOT:
                b = bullet.Bullet(e, p, e)
                bullets.add(b)

            if event.type == QUIT:
                run = False

        clock.tick(30)

        score += env.score

        p.update(p_action)

        env.render()
        e.render()
        p.render()

        clearance_x = []
        clearance_y = []

        clearance_x.clear()
        clearance_y.clear()

        for b in bullets:
            b.update()
            clearance_x.append(b.x)
            clearance_y.append(b.y)

        # Clearance Data
        clearance_x_norm = np.interp(clearance_x, [0, 1000], [0, 1])
        clearance_y_norm = np.interp(clearance_y, [0, 1000], [0, 1])

        # X pos of agent
        x_enemy_norm = np.interp(e.x, [0, 1000], [0, 1])

        env.observation = tf.Variable([x_enemy_norm])

        ai_action = env.choose_action(env.observation)

        if frame_buffer == 2:

            observation_, reward, done, info = env.step(ai_action)
            score += reward
            env.learn(env.observation, reward, observation_, done)
            env.observation = observation_

            frame_buffer = 0

        else:

            frame_buffer += 1

        player_hp = font.render("Your Health: " + str(p.hp), False, (255, 255, 255))
        enemy_hp = font.render("Enemy Health: " + str(e.hp), False, (255, 255, 255))
        level_text = font.render("Level: " + str(config.level), False, (255, 255, 255))
        config.window.blit(player_hp, player_hp_rect)
        config.window.blit(enemy_hp, enemy_hp_rect)
        config.window.blit(level_text, level_text_rect)

        pygame.display.update()

    pygame.quit()
