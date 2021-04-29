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

config.window_height = 800
config.window_width = 1600
config.level = 1


# Helper Function to Write Text Onto Scene
def write(text, location, text_color=(255, 255, 255)):
    textBox = config.reg_font.render(text, False, (0, 0, 0))
    textRect = textBox.get_rect()
    x, y = location
    textRect.center = (x, y)

    renderedText = config.reg_font.render(text, False, text_color)
    config.window.blit(renderedText, textRect)


def reset_game():
    run_ = True

    # Init Player
    p_ = player.Player()

    # Init Enemy
    e_ = enemy.Enemy()

    # Group for bullets
    bullets_ = pygame.sprite.Group()

    # Init Environment for AI to learn
    env_ = environment.Environment(e_, p_, bullets_)

    score_ = 0

    # X pos of agent
    init_x_enemy_norm_ = np.interp(e_.x, [0, 1000], [0, 1])

    # X pos of player
    init_x_player_norm_ = np.interp(p_.x, [0, 1000], [0, 1])

    # Set the state to X pos of player and enemy
    my_tensor_ = tf.constant([[init_x_enemy_norm_, init_x_player_norm_]])
    my_variable_ = tf.Variable(my_tensor_, dtype=np.float64)

    config.level = 1

    return run_, p_, e_, bullets_, env_, score_, my_variable_


if __name__ == '__main__':

    # Game Loop
    run = False
    menu = True

    # Game Initializations
    pygame.init()

    # Initialize Font for Text
    config.reg_font = pygame.font.Font('freesansbold.ttf', 15)
    config.headline_font = pygame.font.Font('freesansbold.ttf', 30)

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
    env = environment.Environment(e, p, bullets)

    # Reward Accumulation
    score = 0

    # Default Actions
    p_action = -1
    ai_action = 0

    # X pos of agent
    init_x_enemy_norm = np.interp(e.x, [0, 1000], [0, 1])

    # X pos of player
    init_x_player_norm = np.interp(p.x, [0, 1000], [0, 1])

    # Set the state to X pos of player and enemy

    my_tensor = tf.constant([[init_x_enemy_norm, init_x_player_norm]])
    my_variable = tf.Variable(my_tensor, dtype=np.float64)
    env.observation = my_variable
    # SHAPE IS (2,1)
    # print(env.observation)

    while menu:

        color = (0, 0, 0)
        config.window.fill(color)

        for event in pygame.event.get():

            if event.type == KEYDOWN:

                if event.key == K_r:
                    done = False
                    run, p, e, bullets, env, score, env.observation = reset_game()

            if event.type == QUIT:
                run = False
                menu = False

        write("Welcome to RL Shootout!", (800, 150), (255, 255, 255))
        write("Left and Right Arrow Keys --- Move", (800, 250), (255, 255, 255))
        write("Spacebar --- Shoot", (800, 300), (255, 255, 255))
        write("Press 'R' to start!", (800, 400), (255, 255, 255))
        # Main Game Loop
        while run:

            color = (0, 0, 0)
            config.window.fill(color)

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
                        # Reset Game
                        run = False

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
                    menu = False

            # Read User Input
            p.update(p_action)

            # Feed the state as input to network to receive action
            ai_action = env.choose_action(env.observation)

            # Every other frame:

            # Input: Action sampled from network's probabilities
            # Output: New state given by action, reward for action, run flag
            observation_, reward, done, info = env.step(ai_action)

            # Append reward
            score += reward

            # Learn based on reward:
            # Input: old state before action, reward, new state after action, run flag
            env.learn(env.observation, reward, observation_, done)

            # Set new state to be old state
            env.observation = observation_

            # If run flag true: Finish
            if done:
                run = False

            # Render text onto screen
            write("Your Health: " + str(p.hp), (800, 750), (255, 255, 255))
            write("Enemy Health: " + str(e.hp), (800, 50), (255, 255, 255))
            write("Level " + str(config.level), (800, 400), (255, 0,  0))

            write("Internal AI Reward: ", (1500, 300), (255, 255, 255))

            write(str(score), (1500, 325), (255, 255, 255))

            pygame.draw.rect(config.window, (255, 255, 255), (200, 0, 5, 800))
            pygame.draw.rect(config.window, (255, 255, 255), (1400, 0, 5, 800))
            pygame.draw.rect(config.window, (255, 255, 255), (200, 100, 1200, 5))
            pygame.draw.rect(config.window, (255, 255, 255), (200, 700, 1200, 5))

            pygame.display.update()

        pygame.display.update()

    pygame.quit()
