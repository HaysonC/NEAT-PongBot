from dummy_neat import dummy_neat

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    """
    return "up" or "down", depending on which way the paddle should go to
    :param paddle_frect: a rectangle representing the coordinates of the paddle
    :param other_paddle_frect: a rectangle representing the opponent paddle
    :param ball_frect: a rectangle representing the ball
    :param table_size: table_size[0], table_size[1] are the dimensions of the table, along the x and the y axis respectively
    :return: "up" or "down", depending on which way the paddle should go to
    """
    if paddle_frect.pos[1] + paddle_frect.size[1] / 2 < ball_frect.pos[1] + ball_frect.size[1] / 2:
        return "down"
    else:
        return "up"


def random_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    """
    return "up" or "down", depending on which way the paddle should go to
    :param paddle_frect: a rectangle representing the coordinates of the paddle
    :param other_paddle_frect: a rectangle representing the opponent paddle
    :param ball_frect: a rectangle representing the ball
    :param table_size: table_size[0], table_size[1] are the dimensions of the table, along the x and the y axis respectively
    :return: "up" or "down", depending on which way the paddle should go to
    """
    import random
    return random.choice(["up", "down"])


class random_dummy_neat(dummy_neat):
    def __init__(self, paddle_size=(10, 70), ball_size=(12, 12), table_size=(400, 480)):
        super().__init__(self._random_ai, paddle_size=paddle_size, ball_size=ball_size, table_size=table_size)

    @staticmethod
    def _random_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
        """
        return "up" or "down", depending on which way the paddle should go to
        :param paddle_frect: a rectangle representing the coordinates of the paddle
        :param other_paddle_frect: a rectangle representing the opponent paddle
        :param ball_frect: a rectangle representing the ball
        :param table_size: table_size[0], table_size[1] are the dimensions of the table, along the x and the y axis respectively
        :return: "up" or "down", depending on which way the paddle should go to
        """
        return random_ai(paddle_frect, other_paddle_frect, ball_frect, table_size)


class chaser_dummy_neat(dummy_neat):
    def __init__(self, paddle_size=(10, 70), ball_size=(12, 12), table_size=(400, 480)):
        super().__init__(self._chaser_ai, paddle_size=paddle_size, ball_size=ball_size, table_size=table_size)

    @staticmethod
    def _chaser_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
        """
        return "up" or "down", depending on which way the paddle should go to
        :param paddle_frect: a rectangle representing the coordinates of the paddle
        :param other_paddle_frect: a rectangle representing the opponent paddle
        :param ball_frect: a rectangle representing the ball
        :param table_size: table_size[0], table_size[1] are the dimensions of the table, along the x and the y axis respectively
        :return: "up" or "down", depending on which way the paddle should go to
        """
        return pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size)


