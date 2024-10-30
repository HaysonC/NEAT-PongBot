
import pygame
import random


from PongAIvAi import fRect
from PongAIvAi import Ball
from PongAIvAi import Paddle
from PongAIvAi import render


class game:
    def __init__(self):
        # Game settings
        self.table_size: tuple[int, int] = (440, 280)
        self.paddle_size: tuple[int, int] = (10, 70)
        self._ball_size: tuple[int, int] = (15, 15)
        self.paddle_speed: int = 1
        self.max_angle: float = 45
        self.paddle_bounce: float = 1.2
        self.wall_bounce: float = 1.00
        self.dust_error: float = 0.00
        self.init_speed_mag: float = 2.00
        self.timeout: float = 0.0003
        self.clock_rate: int = 80
        self.turn_wait_rate: int = 3
        self.score_to_win: int = 5

        # Initialize paddles
        self._paddle = self._Paddle(
            (20, self.table_size[1] / 2),
            self.paddle_size,
            self.paddle_speed,
            self.max_angle,
            1,
            self.timeout
        )
        self._other_paddle = self._Paddle(
            (self.table_size[0] - 20, self.table_size[1] / 2),
            self.paddle_size,
            self.paddle_speed,
            self.max_angle,
            -1,
            self.timeout
        )

        # Initialize move
        self.move = None

        self._ball = Ball(
            (self.table_size[0] / 2, self.table_size[1] / 2),
            self._ball_size,
            self.init_speed_mag,
            self.paddle_bounce,
            self.wall_bounce,
            self.dust_error
        )

        self.score = [0, 0]

    def play(self, move):
        """
        The first player plays

        :param move: "up" or "down"
        :return: None
        """
        self._paddle.set_move(move)
        self._paddle.move(self._other_paddle.frect, self._ball.frect, self.table_size)

    def other_play(self, move):
        """
        The second player plays

        :param move: "up" or "down"
        :return: None
        """
        self._other_paddle.set_move(move)
        self._other_paddle.move(self._paddle.frect, self._ball.frect, self.table_size)

    def update(self):
        """
        Update the game state by simulating the game loop.
        """
        # Move paddles
        self._paddle.move(self._other_paddle.frect, self._ball.frect, self.table_size)
        self._other_paddle.move(self._paddle.frect, self._ball.frect, self.table_size)

        # Move _ball
        inv_move_factor = int((self._ball.speed[0] ** 2 + self._ball.speed[1] ** 2) ** .5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                self._ball.move([self._paddle, self._other_paddle], self.table_size, 1. / inv_move_factor)
        else:
            self._ball.move([self._paddle, self._other_paddle], self.table_size, 1)

        # Check for scoring
        if self._ball.frect.pos[0] + self._ball.size[0] / 2 < 0:
            self.score[1] += 1
            self._ball = Ball(self.table_size, self._ball_size, self.init_speed_mag, self.paddle_bounce, self.wall_bounce, self.dust_error)



    def get_game_state(self):
        return self._paddle.frect, self._other_paddle.frect, self._ball.frect, self.table_size

    class _Paddle(Paddle):
        def __init__(self, pos, size, speed, max_angle, facing, timeout):
            super().__init__(pos, size, speed, max_angle, facing, timeout)
            self.frect = fRect((pos[0] - size[0] / 2, pos[1] - size[1] / 2), size)
            self.speed = speed
            self.size = size
            self.facing = facing
            self.max_angle = max_angle
            self.timeout = timeout
            self.move_getter = lambda _1, _2, _3, _4: None

        def set_move(self, move):
            self.move_getter = lambda _1, _2, _3, _4: move



def main():
    pygame.init()
    screen = pygame.display.set_mode((440, 280))
    pygame.display.set_caption('PongAIvAI Test')

    game_instance = game()

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
        render(screen, [game_instance._paddle, game_instance._other_paddle], game_instance._ball, game_instance.score, game_instance.table_size)
        pygame.display.flip()

        clock.tick(60)


    pygame.quit()


if __name__ == "__main__":
    main()