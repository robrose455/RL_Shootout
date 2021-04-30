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
    K_r,
    K_1,
    K_2,
    K_3,
    K_4

)

run = False
menu = True

reward_mode = 0

# Custom Events
ENEMY_SHOOT = pygame.USEREVENT + 1
RESET = pygame.USEREVENT + 2

config.window_height = 800
config.window_width = 1600
config.level = 1


# Helper Function to Write Text Onto Scene
def write(text, location, text_color, font_type):
    if font_type == 0:
        textBox = config.headline_font.render(text, False, (0, 0, 0))
        textRect = textBox.get_rect()
        x, y = location
        textRect.center = (x, y)

        renderedText = config.headline_font.render(text, False, text_color)
        config.window.blit(renderedText, textRect)

    if font_type == 1:
        textBox = config.reg_font.render(text, False, (0, 0, 0))
        textRect = textBox.get_rect()
        x, y = location
        textRect.center = (x, y)

        renderedText = config.reg_font.render(text, False, text_color)
        config.window.blit(renderedText, textRect)


def render_menu_text():
    write("Welcome to RL Shootout!", (800, 150), (255, 255, 255), 0)
    write("Left and Right Arrow Keys --- Move", (800, 250), (255, 255, 255), 1)
    write("Spacebar --- Shoot", (800, 300), (255, 255, 255), 1)

    write("Choose a Reward Structure:", (800, 400), (255, 255, 255), 0)
    write("(1) Simple Left", (800, 450), (255, 255, 255), 1)
    write("(2) Simple Right", (800, 500), (255, 255, 255), 1)
    write("(3) Stay Close to the Player", (800, 550), (255, 255, 255), 1)
    write("(4) Standard Shootout - Unstable", (800, 600), (255, 255, 255), 1)

    write("Capstone Demo of Reinforcement Learning in Games", (800, 725), (255, 255, 255), 1)
    write("By Rob Rose", (800, 750), (255, 255, 255), 1)


def render_main_text():
    # HP Bar Surface
    pygame.draw.rect(config.window, env.enemy_hp_bar, (200, 0, 1200, 100))
    pygame.draw.rect(config.window, env.player_hp_bar, (200, 700, 1200, 100))

    # Text Displayed on Center
    write("Your Health: " + str(p.hp), (800, 750), (255, 255, 255), 0)
    write("Enemy Health: " + str(e.hp), (800, 50), (255, 255, 255), 0)
    write("Level " + str(config.level), (800, 400), (255, 0, 0), 0)

    # Text Displayed on Left
    write("Controls", (100, 100), (255, 255, 255), 0)
    write("Left & Right: Arrow Keys", (100, 150), (255, 255, 255), 1)
    write("Shoot: Spacebar", (100, 175), (255, 255, 255), 1)
    write("Reset Game: R", (100, 200), (255, 255, 255), 1)

    write("Reward Structure Used:", (100, 300), (255, 255, 255), 1)

    if env.reward_mode == 1:
        write("Simple Left", (100, 350), (255, 255, 255), 1)
        write("* STABLE *", (100, 375), (0, 255, 0), 1)
    if env.reward_mode == 2:
        write("Simple Right", (100, 350), (255, 255, 255), 1)
        write("* STABLE *", (100, 375), (0, 255, 0), 1)
    if env.reward_mode == 3:
        write("Stay Close", (100, 350), (255, 255, 255), 1)
        write("* UNSTABLE *", (100, 375), (255, 255, 0), 1)
    if env.reward_mode == 4:
        write("Standard Shootout", (100, 350), (255, 255, 255), 1)
        write("* UNSTABLE *", (100, 375), (255, 0, 0), 1)

    # Text Displayed On Right
    write("AI Data", (1500, 50), (255, 255, 255), 0)
    write("Enemy Probabilities: ", (1500, 100), (255, 255, 255), 1)
    write("Left: " + str(env.prob_left) + "%", (1500, 150), (255, 255, 255), 1)
    write("Right: " + str(env.prob_right) + "%", (1500, 175), (255, 255, 255), 1)
    write("Shoot: " + str(env.prob_shoot) + "%", (1500, 200), (255, 255, 255), 1)

    write("Steps Taken: ", (1500, 250), (255, 255, 255), 1)
    write(str(env.step_count), (1500, 275), (255, 255, 255), 1)

    write("Learning Rate: " + str(env.nn.learning_rate), (1500, 400), (255, 255, 255), 1)
    write("X Position: " + str(env.host.x), (1500, 425), (255, 255, 255), 1)

    reward_color = (0, 255, 0)
    if score < 0:
        reward_color = (255, 0, 0)

    write("Internal AI Reward: ", (1500, 300), (255, 255, 255), 1)
    write(str(score), (1500, 325), reward_color, 1)

    # Borders
    pygame.draw.rect(config.window, (255, 255, 255), (200, 0, 5, 800))
    pygame.draw.rect(config.window, (255, 255, 255), (1400, 0, 5, 800))
    pygame.draw.rect(config.window, (255, 255, 255), (200, 100, 1200, 5))
    pygame.draw.rect(config.window, (255, 255, 255), (200, 700, 1200, 5))


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

        render_menu_text()

        for event in pygame.event.get():

            if event.type == KEYDOWN:

                if event.key == K_1:
                    done = False
                    run, p, e, bullets, env, score, env.observation = reset_game()
                    env.reward_mode = 1

                if event.key == K_2:
                    done = False
                    run, p, e, bullets, env, score, env.observation = reset_game()
                    env.reward_mode = 2

                if event.key == K_3:
                    done = False
                    run, p, e, bullets, env, score, env.observation = reset_game()
                    env.reward_mode = 3

                if event.key == K_4:
                    done = False
                    run, p, e, bullets, env, score, env.observation = reset_game()
                    env.reward_mode = 4

            if event.type == RESET:

                color = (0, 0, 0)
                config.window.fill(color)
                write("You Lost...", (800, 400), (255, 0, 0), 0)
                pygame.display.update()
                pygame.time.wait(3000)

                done = False
                run, p, e, bullets, env, score, env.observation = reset_game()

            if event.type == QUIT:
                run = False
                menu = False

        # Main Game Loop
        while run:

            color = (0, 0, 0)
            config.window.fill(color)

            pygame.draw.rect(config.window, (74, 71, 71), (200, 50, 1200, 700))

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

            render_main_text()

            pygame.display.update()

        pygame.display.update()

    pygame.quit()
