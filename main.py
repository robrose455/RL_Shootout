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


# Helper Function to Write Text Onto Scene
def write(text, location, text_color=(255, 255, 255)):
    textBox = config.font.render(text, False, (0, 0, 0))
    textRect = textBox.get_rect()
    x, y = location
    textRect.center = (x, y)

    renderedText = config.font.render(text, False, text_color)
    config.window.blit(renderedText, textRect)


if __name__ == '__main__':

    # Game Loop
    run = False
    menu = True

    # Game Initializations
    pygame.init()

    # Initialize Font for Text
    config.font = pygame.font.Font('freesansbold.ttf', 15)

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
    init_x_enemy_norm = np.interp(500, [0, 1000], [0, 1])

    # X pos of player
    init_x_player_norm = np.interp(400, [0, 1000], [0, 1])

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
                    run = True
                    done = False
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
                        "Showing circle"
                        pygame.draw.circle(config.window, (0, 255, 0), (100, 100), 500)

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

            # Update Bullet Clearance
            for b in bullets:
                b.update()

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
            write("Your Health: " + str(p.hp), (100, 450), (255, 255, 255))
            write("Enemy Health: " + str(e.hp), (100, 50), (255, 255, 255))
            write("Level: " + str(config.level), (100, 250), (255, 255, 255))
            write("Internal AI Reward: " + str(score), (100, 300), (255, 255, 255))

            pygame.display.update()

    pygame.quit()
