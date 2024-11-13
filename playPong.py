import sys
import time
from typing import Callable, Optional
import pygame
from typing_extensions import override

import neat_inference
from chaser_ai import chaser_ai

from PongAIvAi import fRect, Ball, Paddle, render
from neat_inference import pong_ai2, pong_ai


class Game(object):
    def __init__(self,
                 table_size: tuple[int, int] = (440, 280),
                 paddle_size: tuple[int, int] = (10, 70),
                 ball_size: tuple[int, int] = (15, 15),
                 paddle_speed: int = 1,
                 max_angle: float = 45,
                 paddle_bounce: float = 1.2,
                 wall_bounce: float = 1.00,
                 dust_error: float = 0.00,
                 init_speed_mag: float = 2.00,
                 timeout: float = 0.0003,
                 clock_rate: int = 80,
                 turn_wait_rate: int = 3,
                 score_to_win: int = 5):
        """
        Initialize the game with default or provided settings.

        :param table_size: Size of the table (width, height)
        :param paddle_size: Size of the paddles (width, height)
        :param ball_size: Size of the ball (width, height)
        :param paddle_speed: Speed of the paddles
        :param max_angle: Maximum angle for paddle deflection
        :param paddle_bounce: Bounce factor for the paddle
        :param wall_bounce: Bounce factor for the wall
        :param dust_error: Error factor for the ball's movement
        :param init_speed_mag: Initial speed magnitude of the ball
        :param timeout: Timeout for paddle movement
        :param clock_rate: Clock rate for the game loop
        :param turn_wait_rate: Wait rate between turns
        :param score_to_win: Score required to win the game
        """
        self._table_size = table_size
        self.paddle_size = paddle_size
        self._ball_size = ball_size
        self.paddle_speed = paddle_speed
        self.max_angle = max_angle
        self.paddle_bounce = paddle_bounce
        self.wall_bounce = wall_bounce
        self.dust_error = dust_error
        self.init_speed_mag = init_speed_mag
        self.timeout = timeout
        self.clock_rate = clock_rate
        self.turn_wait_rate = turn_wait_rate
        self.score_to_win = score_to_win

        # Initialize paddles
        self._paddle = self._Paddle(
            (20, self._table_size[1] // 2),
            self.paddle_size,
            self.paddle_speed,
            self.max_angle,
            1,
            self.timeout
        )
        self._other_paddle = self._Paddle(
            (self._table_size[0] - 20, self._table_size[1] // 2),
            self.paddle_size,
            self.paddle_speed,
            self.max_angle,
            0,
            self.timeout
        )

        # Initialize move
        self.move = None

        self._ball = self._Ball(
            self._table_size,
            self._ball_size,
            self.paddle_bounce,
            self.wall_bounce,
            self.dust_error,
            self.init_speed_mag
        )

        self.score = [0, 0]

    def play(self, move: Optional[str] = None) -> None:
        """
        The first player plays.

        :param move: "up" or "down"
        :return: None
        """
        self._paddle.set_move(move)
        self._paddle.move(self._other_paddle.frect, self._ball.frect, self._table_size)

    def other_play(self, move: Optional[str] = None) -> None:
        """
        The second player plays.

        :param move: "up" or "down"
        :return: None
        """
        self._other_paddle.set_move(move)
        self._other_paddle.move(self._paddle.frect, self._ball.frect, self._table_size)

    def update(self) -> int:
        """
        Update the game state by simulating the game loop.

        :return: 0 if no one scores;
                 1 if the first player scores;
                 2 if the first player wins;
                 3 if the second player hits the ball;
                 -1 if the other player scores;
                 -2 if the other player wins;
                -3 if the first player hits the ball
        """

        # Move paddles
        self._paddle.move(self._other_paddle.frect, self._ball.frect, self._table_size)
        self._other_paddle.move(self._paddle.frect, self._ball.frect, self._table_size)

        # Move ball
        ball = 0
        inv_move_factor = int((self._ball.speed[0] ** 2 + self._ball.speed[1] ** 2) ** .5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                _ = self._ball.move([self._paddle, self._other_paddle], self._table_size, 1. / inv_move_factor)
                ball = _ if _ != 0 else ball
        else:
            ball = self._ball.move([self._paddle, self._other_paddle], self._table_size, 1)
        ret = 0
        # check for hitting the ball
        if ball != 1:
            ret = 3 * ball
        # Check for scoring
        if self._ball.frect.pos[0] + self._ball.size[0] / 2 < 0:
            ret = -1
            self.score[1] += 1
            self._ball = self._Ball(
                self._table_size,
                self._ball_size,
                self.paddle_bounce,
                self.wall_bounce,
                self.dust_error,
                self.init_speed_mag
            )
        elif self._ball.frect.pos[0] + self._ball.size[0] / 2 >= self._table_size[0]:
            ret = 1
            self.score[0] += 1
            self._ball = self._Ball(
                self._table_size,
                self._ball_size,
                self.paddle_bounce,
                self.wall_bounce,
                self.dust_error,
                self.init_speed_mag
            )

        # Check for win
        if self.score[0] >= self.score_to_win or self.score[1] >= self.score_to_win:
            ret = 2 if self.score[0] > self.score[1] else -2
            self.score = [0, 0]
            self._ball = self._Ball(
                self._table_size,self._ball_size,
                self.paddle_bounce,
                self.wall_bounce,
                self.dust_error,
                self.init_speed_mag
            )
        return ret

    def get_game_state(self, side: 1 |- 1 = 1) -> tuple[fRect, fRect, fRect, tuple[int, int]]:
        """
        Get the current game state.

        :return: Tuple containing the paddle, other paddle, ball, and table size
        """
        if side == -1:
            ball = self._Ball(
                self._table_size,
                self._ball_size,
                self.paddle_bounce,
                self.wall_bounce,
                self.dust_error,
                self.init_speed_mag
            )
            ball.frect.pos = (self._table_size[0] - self._ball.frect.pos[0], self._ball.frect.pos[1])
            return self._other_paddle.frect, self._paddle.frect, ball.frect, self._table_size
        else:
            return self._paddle.frect, self._other_paddle.frect, self._ball.frect, self._table_size

    class _Paddle(Paddle):
        def __init__(self, pos: tuple[int, int], size: tuple[int, int], speed: int, max_angle: float, facing: int, timeout: float):
            """
            Initialize the paddle.

            :param pos: Position of the paddle (x, y)
            :param size: Size of the paddle (width, height)
            :param speed: Speed of the paddle
            :param max_angle: Maximum angle for paddle deflection
            :param facing: Direction the paddle is facing
            :param timeout: Timeout for paddle movement
            """
            super().__init__(pos, size, speed, max_angle, facing, timeout)
            self.frect = fRect((pos[0] - size[0] / 2, pos[1] - size[1] / 2), size)
            self.speed = speed
            self.size = size
            self.facing = facing
            self.max_angle = max_angle
            self.timeout = timeout
            # move getter is not an actual getter, it's just a mirror to circumvent
            # the structure of the game
            self.move_getter = lambda _1, _2, _3, _4: None

        def set_move(self, move: str | int) -> None:
            """
            Set the move for the paddle.

            :param move: "up" or "down", or -1 (d), 0, 1 (u) (for AI)
            :return: None
            """
            if move == -1:
                move = "down"
            elif move == 1:
                move = "up"
            elif move == 0:
                move = None

            self.move_getter = lambda _1, _2, _3, _4: move

        def get_move(self) -> Optional[str]:
            """
            Get the move for the paddle.

            :return: "up" or "down" or None
            """
            return self.move_getter(None, None, None, None)

    class _Ball(Ball):
        def __init__(self, table_size: tuple[int, int], size: tuple[int, int], paddle_bounce: float, wall_bounce: float,
                     dust_error: float, init_speed_mag: float):
            super().__init__(table_size, size, paddle_bounce, wall_bounce, dust_error, init_speed_mag)

        @override
        def move(self, paddles
                 , table_size,
                 move_factor) -> 1 | 0 | -1:
            super().move(paddles, table_size, move_factor)
            ret = 0
            for paddle in paddles:
                if not ((paddle.facing == 1 and self.get_center()[0] < paddle.frect.pos[0] + paddle.frect.size[
                    0] / 2) or
                        (paddle.facing == 0 and self.get_center()[0] > paddle.frect.pos[0] + paddle.frect.size[0] / 2)):
                    ret = paddle.facing
            return ret


    # Getters
    def get_ball(self) -> _Ball:
        """
        Get the ball object.

        :return: _Ball object
        """
        return self._ball

    def get_paddle(self) -> _Paddle:
        """
        Get the first paddle object.

        :return: _Paddle object
        """
        return self._paddle

    def get_other_paddle(self) -> _Paddle:
        """
        Get the second paddle object.

        :return: _Paddle object
        """
        return self._other_paddle

    def get_table_size(self) -> tuple[int, int]:
        """
        Get the table size.

        :return: Tuple containing the table size (width, height)
        """
        return self._table_size

    def render(self):
        """
        Render the game state.

        :return: None
        """
        from PongAIvAi import render
        render(pygame.display.set_mode(self._table_size),[self._paddle, self._other_paddle], self._ball, self.score, self._table_size)


pygame.init()
class HumanPlayer():
    def __init__(self, up: str = "UP", down: str = "DOWN"):
        """
        Initialize the human player.

        For arrow keys, use "UP", "DOWN", "LEFT", "RIGHT"

        :param up: The key to move the paddle up.
        :param down: The key to move the paddle down.
        """
        self.up: str = up
        self.down: str = down


    def __call__(self, *args) -> Optional[str]:
        """
        Get the move from the human player.

        :return: "up" or "down" or None
        """
        keys = pygame.key.get_pressed()

        if keys[eval(f"pygame.K_{self.up}")]:
            return "up"
        elif keys[eval(f"pygame.K_{self.down}")]:
            return "down"
        else:
            return None


def visualize_game_loop(game_instance: Game,
                        player1: Callable = chaser_ai,
                        player2: Callable = chaser_ai,
                        caption: str = "Pong",
                        test_reward: bool = False,
                        tickTime: float = 0.06) -> None:
    """
    Visualize the game loop.

    :param game_instance: The game instance to visualize.
    :param player1: The first player.
    :param player2: The second player.
    :param caption: The caption for the game window.
    :param test_reward: Whether to test the reward.
    :param tickTime: The tick time for the game loop.
    :return: None
    """

    clock = pygame.time.Clock()

    screen = pygame.display.set_mode(game_instance.get_table_size())
    pygame.display.set_caption(caption)

    font = pygame.font.Font(None, 36)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        game_instance.play(player1(*game_instance.get_game_state(side = 1)))
        game_instance.other_play(player2(*game_instance.get_game_state(side = -1)))
        if test_reward:
            from neat_train import update_fitness
            class _genome:
                def __init__(self):
                    self.fitness: float = 0
            myDummyGenome = _genome()
            otherDummyGenome = _genome()
            update_fitness(game_instance, myDummyGenome, otherDummyGenome)
            if myDummyGenome.fitness != 0 or otherDummyGenome.fitness != 0:
                # print the finess on the pygame screen
                text = font.render(f"Fitness: {myDummyGenome.fitness} {otherDummyGenome.fitness}", True, (255, 255, 255))
                screen.blit(text, (10, 10))

        else :
            game_instance.update()
        pygame.display.flip()
        render(screen, [game_instance.get_paddle(), game_instance.get_other_paddle()], game_instance.get_ball(), game_instance.score, game_instance.get_table_size())
        clock.tick(1/tickTime)

def random_play(game_instance: Game, n:int) -> float:
    """
    Play the game n times and return the time taken.

    :param game_instance: The game instance to play.
    :param n: The number of times to play the game.
    :return: The time taken to play the game n times.
    """
    i = 0
    prev = time.time()
    while i < n:
        game_instance.play()
        game_instance.other_play()
        res = game_instance.update()
        if res != 0:
            i += 1
    timeTaken = time.time() - prev
    return timeTaken



if __name__ == '__main__':
    # import neat_inference
    visualize_game_loop(Game(),pong_ai,HumanPlayer(),
                        tickTime=0.01)