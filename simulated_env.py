# PongAIvAI - Flexible Simulation and Visualization
# Authors: Michael Guerzhoy and Denis Begun, 2014-2022.
# Modified to include optional rendering and clock control.

import os
import sys
import pygame
import random
import math
import logging
import time
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
# ----------------------------
# Rectangle Class with Floating-Point Precision
# ----------------------------

class fRect:
    '''
    pygame's Rect class can only represent whole integer vertices,
    so we create a rectangle class that can have floating-point coordinates.
    '''

    def __init__(self, pos, size):
        self.pos = (pos[0], pos[1])    # (x, y) position
        self.size = (size[0], size[1])  # (width, height)

    def move(self, x, y):
        '''
        Returns a new fRect moved by (x, y) without altering the original.
        '''
        return fRect((self.pos[0] + x, self.pos[1] + y), self.size)

    def move_ip(self, x, y, move_factor=1):
        '''
        Moves the rectangle in place by (x, y), optionally scaled by move_factor.
        '''
        self.pos = (self.pos[0] + x * move_factor, self.pos[1] + y * move_factor)

    def get_rect(self):
        '''
        Converts the fRect to Pygame's Rect with integer values for compatibility.
        '''
        return pygame.Rect(int(self.pos[0]), int(self.pos[1]), int(self.size[0]), int(self.size[1]))

    def copy(self):
        '''
        Creates a copy of the fRect.
        '''
        return fRect(self.pos, self.size)

    def intersect(self, other_frect):
        '''
        Checks if this rectangle intersects with another fRect.

        :param other_frect: Another instance of fRect.
        :return: True if rectangles intersect, False otherwise.
        '''
        # Two rectangles intersect if their projections on both axes overlap
        for i in range(2):
            if self.pos[i] < other_frect.pos[i]:
                if other_frect.pos[i] >= self.pos[i] + self.size[i]:
                    return False
            elif self.pos[i] > other_frect.pos[i]:
                if self.pos[i] >= other_frect.pos[i] + other_frect.size[i]:
                    return False
        return True  # Rectangles intersect

# ----------------------------
# Paddle Class
# ----------------------------

class Paddle:
    def __init__(self, pos, size, speed, max_angle, facing, timeout):
        self.frect = fRect((pos[0] - size[0] / 2, pos[1] - size[1] / 2), size)
        self.speed = speed
        self.size = size
        self.facing = facing  # 1 for left, 0 for right
        self.max_angle = max_angle
        self.timeout = timeout
        self.move_getter = None  # Function to decide movement

    def factor_accelerate(self, factor):
        '''
        Scales the paddle's speed by the given factor.
        '''
        self.speed *= factor


    def move(self, enemy_frect, ball_frect, table_size, dt=1):
        '''
        Moves the paddle based on the move_getter function.

        :param enemy_frect: fRect of the opposing paddle.
        :param ball_frect: fRect of the ball.
        :param table_size: Tuple (width, height) of the game table.
        :param dt: Delta time multiplier for movement (default=1).
        '''
        direction = self.move_getter(self.frect.copy(), enemy_frect.copy(), ball_frect.copy(), tuple(table_size))
        if direction == "up":
            self.frect.move_ip(0, -self.speed * dt)
        elif direction == "down":
            self.frect.move_ip(0, self.speed * dt)
        # If direction is "stay" or None, do not move

        # Keep paddle within the screen bounds
        to_bottom = (self.frect.pos[1] + self.frect.size[1]) - table_size[1]
        if to_bottom > 0:
            self.frect.move_ip(0, -to_bottom)
        to_top = self.frect.pos[1]
        if to_top < 0:
            self.frect.move_ip(0, -to_top)

    def get_angle(self, y):
        '''
        Calculates the reflection angle based on where the ball hits the paddle.

        :param y: Y-coordinate of the ball's center.
        :return: Angle in radians.
        '''
        center = self.frect.pos[1] + self.size[1] / 2
        rel_dist_from_center = ((y - center) / self.size[1])
        rel_dist_from_center = max(-0.5, min(0.5, rel_dist_from_center))  # Clamp between -0.5 and 0.5
        sign = 1 - 2 * self.facing  # 1 if left paddle, -1 if right paddle

        return sign * rel_dist_from_center * self.max_angle * math.pi / 180

# ----------------------------
# Ball Class
# ----------------------------

class Ball:
    def __init__(self, table_size, size, paddle_bounce, wall_bounce, dust_error, init_speed_mag):
        rand_ang = (.4 + .4 * random.random()) * math.pi * (1 - 2 * (random.random() > .5)) + .5 * math.pi
        speed = (init_speed_mag * math.cos(rand_ang), init_speed_mag * math.sin(rand_ang))
        pos = (table_size[0] / 2, table_size[1] / 2)
        self.frect = fRect((pos[0] - size[0] / 2, pos[1] - size[1] / 2), size)
        self.speed = speed
        self.size = size
        self.paddle_bounce = paddle_bounce
        self.wall_bounce = wall_bounce
        self.dust_error = dust_error
        self.init_speed_mag = init_speed_mag
        self.prev_bounce = None

    def get_center(self):
        '''
        Returns the center coordinates of the ball.
        '''
        return (self.frect.pos[0] + 0.5 * self.frect.size[0], self.frect.pos[1] + 0.5 * self.frect.size[1])

    def get_speed_mag(self):
        '''
        Returns the magnitude of the ball's speed.
        '''
        return math.sqrt(self.speed[0] ** 2 + self.speed[1] ** 2)

    def factor_accelerate(self, factor):
        '''
        Scales the ball's speed by the given factor.
        '''
        self.speed = (self.speed[0] * factor, self.speed[1] * factor)

    def move(self, paddles, table_size, move_factor):
        '''
        Moves the ball and handles collisions with walls and paddles.

        :param paddles: List of Paddle instances.
        :param table_size: Tuple (width, height) of the game table.
        :param move_factor: Multiplier for movement, can be used for time-based movement.
        '''
        moved = False
        walls_Rects = [
            pygame.Rect(-100, -100, table_size[0] + 200, 100),
            pygame.Rect(-100, table_size[1], table_size[0] + 200, 100)
        ]

        # Collision with walls
        for wall_rect in walls_Rects:
            if self.frect.get_rect().colliderect(wall_rect):
                # Backtrack until out of wall
                while self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(-0.1 * self.speed[0], -0.1 * self.speed[1], move_factor)
                # Apply wall bounce with possible dust error
                r1 = 1 + 2 * (random.random() - 0.5) * self.dust_error
                r2 = 1 + 2 * (random.random() - 0.5) * self.dust_error

                self.speed = (self.wall_bounce * self.speed[0] * r1, -self.wall_bounce * self.speed[1] * r2)
                # Move the ball forward with new speed until out of wall
                while self.frect.get_rect().colliderect(wall_rect):
                    self.frect.move_ip(0.1 * self.speed[0], 0.1 * self.speed[1], move_factor)
                moved = True

        # Collision with paddles
        for paddle in paddles:
            if self.frect.intersect(paddle.frect):
                # Prevent multiple bounces from the same paddle
                if paddle is self.prev_bounce:
                    continue

                # Backtrack until out of paddle
                while self.frect.intersect(paddle.frect):
                    self.frect.move_ip(-0.1 * self.speed[0], -0.1 * self.speed[1], move_factor)

                theta = paddle.get_angle(self.frect.pos[1] + 0.5 * self.frect.size[1])
                v = self.speed

                # Rotate velocity by theta
                v_rot = [
                    math.cos(theta) * v[0] - math.sin(theta) * v[1],
                    math.sin(theta) * v[0] + math.cos(theta) * v[1]
                ]

                # Reverse x direction
                v_rot[0] = -v_rot[0]

                # Rotate back
                v_new = [
                    math.cos(-theta) * v_rot[0] - math.sin(-theta) * v_rot[1],
                    math.sin(-theta) * v_rot[0] + math.cos(-theta) * v_rot[1]
                ]

                # Ensure minimum speed in x direction
                if v_new[0] * (2 * paddle.facing - 1) < 1:
                    v_new[1] = (v_new[1] / abs(v_new[1])) * math.sqrt(v_new[0] ** 2 + v_new[1] ** 2 - 1)
                    v_new[0] = 2 * paddle.facing - 1

                # Update speed with paddle bounce
                self.speed = (v_new[0] * self.paddle_bounce, v_new[1] * self.paddle_bounce)
                self.prev_bounce = paddle

                # Move the ball forward with new speed until out of paddle
                while self.frect.intersect(paddle.frect):
                    self.frect.move_ip(0.1 * self.speed[0], 0.1 * self.speed[1], move_factor)

                moved = True

        # Move the ball if no collision occurred
        if not moved:
            self.frect.move_ip(self.speed[0], self.speed[1], move_factor)

# ----------------------------
# Player Movement Functions
# ----------------------------

def directions_from_input(paddle_rect, other_paddle_rect, ball_rect, table_size):
    '''
    Placeholder for player1 movement based on keyboard input.
    Since we may run in headless mode, return "stay" by default.
    '''
    # In headless mode, keyboard input is not applicable. Replace with AI or default behavior.
    return "stay"

def chaser_ai_move(paddle_rect, enemy_paddle_rect, ball_rect, table_size):
    '''
    Simple AI that moves the paddle towards the ball's y position.
    '''
    paddle_center = paddle_rect.pos[1] + paddle_rect.size[1] / 2
    ball_y = ball_rect.pos[1] + ball_rect.size[1] / 2

    if paddle_center < ball_y:
        return "down"
    elif paddle_center > ball_y:
        return "up"
    else:
        return "stay"

# ----------------------------
# Timeout Function
# ----------------------------

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''
    Executes a function with a timeout. If the function does not complete within
    the specified timeout_duration, it returns the default value.

    :param func: Function to execute.
    :param args: Positional arguments for the function.
    :param kwargs: Keyword arguments for the function.
    :param timeout_duration: Maximum time (in seconds) to allow the function to run.
    :param default: Default value to return if timeout occurs.
    :return: Function's result or default value.
    '''
    import threading

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        print("TIMEOUT")
        return default
    else:
        return it.result

# ----------------------------
# Game Mechanics Functions
# ----------------------------

def check_point(score, ball, table_size):
    '''
    Checks if a point has been scored and updates the score.

    :param score: List [left_score, right_score].
    :param ball: Ball instance.
    :param table_size: Tuple (width, height) of the game table.
    :return: Tuple (ball, score).
    '''
    if ball.frect.pos[0] + ball.size[0] / 2 < 0:
        score[1] += 1
        ball = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
        return (ball, score)
    elif ball.frect.pos[0] + ball.size[0] / 2 >= table_size[0]:
        ball = Ball(table_size, ball.size, ball.paddle_bounce, ball.wall_bounce, ball.dust_error, ball.init_speed_mag)
        score[0] += 1
        return (ball, score)

    return (ball, score)

# ----------------------------
# Initialization Function
# ----------------------------

def init_pygame(display, table_size):
    '''
    Initializes Pygame and sets up the display window based on the display flag.

    :param display: Boolean flag to determine if rendering should be enabled.
    :param table_size: Tuple (width, height) of the game table.
    :return: Pygame screen object if display=True, else None.
    '''
    if display:
        pygame.display.init()
        screen = pygame.display.set_mode(table_size)
        pygame.display.set_caption('PongAIvAI - Demo')
    else:
        # Set SDL to use the dummy video driver for headless mode
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        screen = None  # No display in headless mode
    return screen

# ----------------------------
# Simulation and Game Loop Functions
# ----------------------------

def render(screen, paddles, ball, score, table_size):
    '''
    Renders the current game state to the screen.

    :param screen: Pygame screen object.
    :param paddles: List of Paddle instances.
    :param ball: Ball instance.
    :param score: List [left_score, right_score].
    :param table_size: Tuple (width, height) of the game table.
    '''
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    screen.fill(BLACK)

    pygame.draw.rect(screen, WHITE, paddles[0].frect.get_rect())
    pygame.draw.rect(screen, WHITE, paddles[1].frect.get_rect())

    pygame.draw.circle(screen, WHITE, (int(ball.get_center()[0]), int(ball.get_center()[1])),
                       int(ball.frect.size[0] / 2), 0)

    pygame.draw.line(screen, WHITE, [screen.get_width() / 2, 0], [screen.get_width() / 2, screen.get_height()])

    score_font = pygame.font.Font("freesansbold.ttf", 32)
    screen.blit(score_font.render(str(score[0]), True, WHITE), [int(0.4 * table_size[0]) - 8, 0])
    screen.blit(score_font.render(str(score[1]), True, WHITE), [int(0.6 * table_size[0]) - 8, 0])

    pygame.display.flip()

def game_loop(screen, paddles, ball, table_size, score_to_win, display, probe_game=False):
    '''
    Runs the main game loop, handling movement, collisions, and rendering.

    :param screen: Pygame screen object or None for headless mode.
    :param paddles: List of Paddle instances.
    :param ball: Ball instance.
    :param table_size: Tuple (width, height) of the game table.
    :param score_to_win: Points needed to win the game.
    :param display: Boolean flag to determine if rendering should be enabled.
    :param probe_game: Boolean flag to determine if the game should be visualized.
    :return: Final score as a list [left_score, right_score].
    '''
    score = [0, 0]

    if display and probe_game:
        clock = pygame.time.Clock()

    while max(score) < score_to_win:
        old_score = score[:]

        # Update ball position and handle collisions
        inv_move_factor = int((ball.speed[0] ** 2 + ball.speed[1] ** 2) ** 0.5)
        if inv_move_factor > 0:
            for _ in range(inv_move_factor):
                ball.move(paddles, table_size, 1. / inv_move_factor)
        else:
            ball.move(paddles, table_size, 1)

        # Update paddles based on player decisions
        paddles[0].move(paddles[1].frect, ball.frect, table_size)
        paddles[1].move(paddles[0].frect, ball.frect, table_size)

        # Check for scoring
        ball, score = check_point(score, ball, table_size)

        if display and probe_game:
            if score != old_score:
                font = pygame.font.Font(None, 32)
                if score[0] != old_score[0]:
                    screen.blit(font.render("Left scores!", True, WHITE, BLACK), [0, 32])
                else:
                    screen.blit(font.render("Right scores!", True, WHITE, BLACK), [int(table_size[0] / 2 + 20), 32])

                pygame.display.flip()
                clock.tick(3)  # Slow down to observe score

            # Render the current game state
            render(screen, paddles, ball, score, table_size)

            # Handle events to prevent the window from freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Control frame rate
            clock.tick(80)

    if display and probe_game:
        # Display the result
        font = pygame.font.Font(None, 64)
        if score[0] > score[1]:
            screen.blit(font.render("Left wins!", True, WHITE, BLACK), [24, 32])
        else:
            screen.blit(font.render("Right wins!", True, WHITE, BLACK), [24, 32])
        pygame.display.flip()
        clock.tick(2)

        # Wait for a short duration before quitting
        pygame.time.delay(2000)

    return score

def simulate_game(player1_move_func, player2_move_func, table_size=(440, 280), score_to_win=5, max_steps=1000, display=False, probe_game=False):
    '''
    Simulates a single game between two players.

    :param player1_move_func: Function to decide movement for player1.
    :param player2_move_func: Function to decide movement for player2.
    :param table_size: Tuple (width, height) of the game table.
    :param score_to_win: Points needed to win the game.
    :param max_steps: Maximum number of steps to prevent infinite games.
    :param display: Boolean flag to determine if rendering should be enabled.
    :param probe_game: Boolean flag to determine if the game should be visualized.
    :return: Final score as a list [left_score, right_score].
    '''
    # Initialize Pygame and the display based on the display flag
    screen = init_pygame(display, table_size)

    # Initialize paddles and ball
    paddle_size = (10, 70)
    paddle_speed = 5
    max_angle = 45
    paddle_bounce = 1.2
    wall_bounce = 1.00
    dust_error = 0.00
    init_speed_mag = 5.00
    timeout = 0.0003

    paddles = [
        Paddle((20, table_size[1] / 2), paddle_size, paddle_speed, max_angle, 1, timeout),
        Paddle((table_size[0] - 20, table_size[1] / 2), paddle_size, paddle_speed, max_angle, 0, timeout)
    ]
    ball = Ball(table_size, (15, 15), paddle_bounce, wall_bounce, dust_error, init_speed_mag)

    paddles[0].move_getter = player1_move_func
    paddles[1].move_getter = player2_move_func

    # Run the game loop
    score = game_loop(screen, paddles, ball, table_size, score_to_win, display, probe_game)

    # Quit Pygame if not in headless mode
    if display:
        pygame.quit()

    return score

# ----------------------------
# Main Function for Testing
# ----------------------------

def test():
    '''
    Main function to test the PongAIvAI simulation with optional rendering.
    '''
    # Define the mode of operation
    # Set probe_game=True for visualization, False for headless simulation

    probe_game = False  # Change to True to visualize the game
    if probe_game:
        print("Running in visualization mode...")
        pygame.init()
        pygame.font.init()
    # Define player movement functions
    player1 = chaser_ai_move
    player2 = chaser_ai_move

    if probe_game:
        # Run a single game with visualization
        print("Running a visual game...")
        score = simulate_game(player1_move_func=player1, player2_move_func=player2,
                             table_size=(440, 280), score_to_win=5, max_steps=1000,
                             display=True, probe_game=True)
        print(f"Game complete. Final Score: Left {score[0]} - Right {score[1]}")
    else:
        # Run multiple headless simulations for testing
        num_simulations = 10
        total_fitness = 0
        print(f"Running {num_simulations} headless simulations...")
        for i in range(num_simulations):
            score = simulate_game(player1_move_func=player1, player2_move_func=player2,
                                 table_size=(440, 280), score_to_win=5, max_steps=1000,
                                 display=False, probe_game=False)
            total_fitness += score[0]
            print(f"Simulation {i+1}: Player1 scored {score[0]} points.")

        average_fitness = total_fitness / num_simulations
        print(f"Average fitness over {num_simulations} simulations: {average_fitness} points.")

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == '__main__':
    test()
