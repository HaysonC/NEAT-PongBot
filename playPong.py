import pygame
import random
from PongAIvAi import fRect, Ball, Paddle, render

class Game:
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

        self._ball = Ball(
            (self._table_size[0] / 2, self._table_size[1] / 2),
            self._ball_size,
            self.paddle_bounce,
            self.wall_bounce,
            self.dust_error,
            self.init_speed_mag
        )

        self.score = [0, 0]

    def play(self, move: str | None) -> None:
        """
        The first player plays.

        :param move: "up" or "down"
        :return: None
        """
        self._paddle.set_move(move)
        self._paddle.move(self._other_paddle.frect, self._ball.frect, self._table_size)

    def other_play(self, move: str | None) -> None:
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

        :return: 1 if the first player scores, 0 if no one scores, -1 if the second player scores
        """
        # Move paddles
        self._paddle.move(self._other_paddle.frect, self._ball.frect, self._table_size)
        self._other_paddle.move(self._paddle.frect, self._ball.frect, self._table_size)

        # Move ball
        inv_move_factor = int((self._ball.speed[0] ** 2 + self._ball.speed[1] ** 2) ** .5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                self._ball.move([self._paddle, self._other_paddle], self._table_size, 1. / inv_move_factor)
        else:
            self._ball.move([self._paddle, self._other_paddle], self._table_size, 1)
        ret = 0
        # Check for scoring
        if self._ball.frect.pos[0] + self._ball.size[0] / 2 < 0:
            ret = -1
            self.score[1] += 1
            self._ball = Ball(
                (self._table_size[0] / 2, self._table_size[1] / 2),
                self._ball_size,
                self.paddle_bounce,
                self.wall_bounce,
                self.dust_error,
                self.init_speed_mag
            )
        elif self._ball.frect.pos[0] + self._ball.size[0] / 2 >= self._table_size[0]:
            ret = 1
            self.score[0] += 1
            self._ball = Ball(
                (self._table_size[0] / 2, self._table_size[1] / 2),
                self._ball_size,
                self.paddle_bounce,
                self.wall_bounce,
                self.dust_error,
                self.init_speed_mag
            )

        # Check for win
        if self.score[0] >= self.score_to_win or self.score[1] >= self.score_to_win:
            self.score = [0, 0]
            self._ball = Ball(
                (self._table_size[0] / 2, self._table_size[1] / 2),
                self._ball_size,
                self.paddle_bounce,
                self.wall_bounce,
                self.dust_error,
                self.init_speed_mag
            )
        return ret

    def get_game_state(self) -> tuple[fRect, fRect, fRect, tuple[int, int]]:
        """
        Get the current game state.

        :return: Tuple containing the paddle, other paddle, ball, and table size
        """
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
            self._move_getter = lambda _1, _2, _3, _4: None

        def set_move(self, move: str) -> None:
            """
            Set the move for the paddle.

            :param move: "up" or "down"
            :return: None
            """
            self._move_getter = lambda _1, _2, _3, _4: move

    def get_ball(self) -> Ball:
        """
        Get the ball object.

        :return: Ball object
        """
        return self._ball

    def get_paddle(self) -> Paddle:
        """
        Get the first paddle object.

        :return: Paddle object
        """
        return self._paddle

    def get_other_paddle(self) -> Paddle:
        """
        Get the second paddle object.

        :return: Paddle object
        """
        return self._other_paddle

    def get_table_size(self) -> tuple[int, int]:
        """
        Get the table size.

        :return: Tuple containing the table size (width, height)
        """
        return self._table_size

def main() -> None:
    """
    Main function to run the game.
    """
    pygame.init()
    screen = pygame.display.set_mode((440, 280))
    pygame.display.set_caption('PongAIvAI Test')

    game_instance = Game()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            game_instance.play("up")
        elif keys[pygame.K_DOWN]:
            game_instance.play("down")
        else:
            game_instance.play(None)

        game_instance.other_play("up" if random.random() > 0.5 else "down")

        game_instance.update()

        screen.fill((0, 0, 0))

        render(screen, [game_instance.get_paddle(), game_instance.get_other_paddle()], game_instance.get_ball(), game_instance.score, game_instance.get_table_size())
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()