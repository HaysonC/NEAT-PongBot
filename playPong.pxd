from PongAIvAi import fRect, Ball, Paddle

cdef class Game:
    def __init__(self,
                 tuple table_size=(440, 280),
                 tuple paddle_size=(10, 70),
                 int paddle_speed=1,
                 float max_angle=45,
                 float paddle_bounce=1.2,
                 float wall_bounce=1.00,
                 float dust_error=0.00,
                 float init_speed_mag=2.00,
                 float timeout=0.0003,
                 int clock_rate=80,
                 int turn_wait_rate=3,
                 int score_to_win=5):
        self._table_size = table_size
        self.paddle_size = paddle_size
        self._ball_size = (15, 15)
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

        self._ball = Ball(
            (self._table_size[0] / 2, self._table_size[1] / 2),
            self._ball_size,
            self.paddle_bounce,
            self.wall_bounce,
            self.dust_error,
            self.init_speed_mag
        )

        self.score = [0, 0]

    def play(self, move: str = None) -> None:
        self._paddle.set_move(move)
        self._paddle.move(self._other_paddle.frect, self._ball.frect, self._table_size)

    def other_play(self, move: str = None) -> None:
        self._other_paddle.set_move(move)
        self._other_paddle.move(self._paddle.frect, self._ball.frect, self._table_size)

    def update(self) -> int:
        self._paddle.move(self._other_paddle.frect, self._ball.frect, self._table_size)
        self._other_paddle.move(self._paddle.frect, self._ball.frect, self._table_size)

        inv_move_factor = int((self._ball.speed[0] ** 2 + self._ball.speed[1] ** 2) ** .5)
        if inv_move_factor > 0:
            for i in range(inv_move_factor):
                self._ball.move([self._paddle, self._other_paddle], self._table_size, 1. / inv_move_factor)
        else:
            self._ball.move([self._paddle, self._other_paddle], self._table_size, 1)
        ret = 0
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

    cdef class _Paddle(Paddle):
        def __init__(self, tuple pos, tuple size, int speed, float max_angle, int facing, float timeout):
            super().__init__(pos, size, speed, max_angle, facing, timeout)
            self.frect = fRect((pos[0] - size[0] / 2, pos[1] - size[1] / 2), size)
            self.speed = speed
            self.size = size
            self.facing = facing
            self.max_angle = max_angle
            self.timeout = timeout
            self.move_getter = lambda _1, _2, _3, _4: None

        def set_move(self, str move) -> None:
            self.move_getter = lambda _1, _2, _3, _4: move