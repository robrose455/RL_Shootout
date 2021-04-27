import environment
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

# Custom Events
ENEMY_SHOOT = pygame.USEREVENT + 1

config.window_height = 500
config.window_width = 1000
config.level = 1

if __name__ == '__main__':

    # Game Loop
    run = True

    # Game Initializations
    pygame.init()

    # Frame rate
    clock = pygame.time.Clock()

    # Set Window Size
    config.window = pygame.display.set_mode((config.window_width, config.window_height))

    # Buffer for AI to learn
    frame_buffer = 0

    # Init Player
    p = player.Player()

    # Init Enemy
    e = enemy.Enemy()

    # Group for bullets
    bullets = pygame.sprite.Group()

    # Init Environment for AI to learn
    env = environment.Environment(e, p)

    # Reward Accumulation
    score = 0

    # Default Actions
    p_action = -1
    ai_action = 0

    # Text to be rendered onto screen
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

    reward_text = font.render(str(score), False, (0, 0, 0))
    reward_text_rect = reward_text.get_rect()
    reward_text_rect.center = (50, 300)

    # X pos of agent
    init_x_enemy_norm = np.interp(500, [0, 1000], [0, 1])

    # X pos of player
    init_x_player_norm = np.interp(400, [0, 1000], [0, 1])

    # Set the state to X pos of player and enemy

    my_tensor = tf.constant([[init_x_enemy_norm], [init_x_player_norm]])
    my_variable = tf.Variable(my_tensor, dtype=np.float64)
    env.observation = my_variable
    #SHAPE IS (2,1)
    #print(env.observation)

    # Main Game Loop
    while run:

        # Define Framerate
        clock.tick(30)

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

        # Read User Input
        p.update(p_action)

        # Update Bullet Clearance
        for b in bullets:
            b.update()

        # Feed the state as input to network to receive action
        ai_action = env.choose_action(env.observation)

        # Every other frame:
        if frame_buffer == 1:

            # Input: Action sampled from network's probabilities
            # Output: New state given by action, reward for action, run flag
            observation_, reward, done, info = env.step(ai_action)

            # Append reward
            score += reward

            # Learn based on reward:
            # Input: old state before action, reward, new state after action, run flag
            print("----------")
            print(env.observation.value())
            print(observation_.value())
            env.learn(env.observation, reward, observation_, done)

            # Set new state to be old state
            env.observation = observation_

            # If run flag true: Finish
            if done:
                run = False

            frame_buffer = 0

        else:

            frame_buffer += 1

        # Render text onto screen
        player_hp = font.render("Your Health: " + str(p.hp), False, (255, 255, 255))
        enemy_hp = font.render("Enemy Health: " + str(e.hp), False, (255, 255, 255))
        level_text = font.render("Level: " + str(config.level), False, (255, 255, 255))
        reward_text = font.render("Internal AI Reward: " + str(score), False, (255, 255, 255))

        # Blit text onto screen
        config.window.blit(player_hp, player_hp_rect)
        config.window.blit(enemy_hp, enemy_hp_rect)
        config.window.blit(level_text, level_text_rect)
        config.window.blit(reward_text, reward_text_rect)

        pygame.display.update()

    pygame.quit()
