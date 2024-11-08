# if we want the agent to train with other types of agenet, we cast it into a "dummy" neat so the same kinds of operations can be performed on it and doesn't break the code.
from typing import Callable

import neat

from PongAIvAi import fRect

class dummy_neat(neat.DefaultGenome):
    def __init__(self, move_getter: Callable,
                 paddle_size: tuple = (10, 50),
                 ball_size: tuple = (10, 10),
                 table_size: tuple = (400, 400)) -> None:
        """
        :param move_getter: a function that returns the move of the agent, so it would be the other type of agent
        :param paddle_size: the size of the paddle
        :param ball_size: the size of the ball
        :param table_size: the size of the table
        """
        super().__init__(1)
        self.fitness = 0
        self.net = self._net(move_getter, paddle_size, ball_size, table_size)
        self.move_getter = move_getter

    def __call__(self, *args, **kwargs):
        return self.move_getter(*args, **kwargs)

    # dummy net class
    class _net:
        def __init__(self, move_getter: Callable,
                     paddle_size: tuple = (10, 50),
                     ball_size: tuple = (10, 10),
                     table_size: tuple = (400, 400)) -> None:
            """
            Dummy net class for the dummy neat class

            :param move_getter: a function that returns the move of the agent, so it would be the other type of agent
            :param paddle_size: the size of the paddle
            :param ball_size: the size of the ball
            :param table_size: the size of the table
            """

            self.move_getter = move_getter
            self._paddle_size = paddle_size
            self._ball_size = ball_size
            self._table_size = table_size


        def activate(self, l: iter) -> iter:
            """
            Activate the dummy  neural network

            :param l: the input to the dummy neural network
            :return: the output of the dummy neural network
            """

            _ball = fRect(pos=(l[0], l[1]), size=self._ball_size)
            _paddle = fRect(pos=(l[2], l[3]), size=self._paddle_size)
            _other_paddle = fRect(pos=(l[2], l[3]), size=self._paddle_size)
            move = self.move_getter(_paddle, _other_paddle, _ball, self._table_size)
            if move is None:
                return [0, 1, 0]
            elif move == "up":
                return [0, 0, 1]
            elif move == "down":
                return [1, 0, 0]
            else:
                return [0, 1, 0]

        def __str__(self):
            return "Dummy net"