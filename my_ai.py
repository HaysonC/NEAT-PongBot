def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    """
    return "up" or "down", depending on which way the paddle should go to
    :param paddle_frect: a rectangle representing the coordinates of the paddle
    :param other_paddle_frect: a rectangle representing the opponent paddle
    :param ball_frect: a rectangle representing the ball
    :param table_size: table_size[0], table_size[1] are the dimensions of the table, along the x and the y axis respectively
    :return: "up" or "down", depending on which way the paddle should go to
    """


    """
         0             x
     |------------->
     |
     |
     |
 y   v
    
    """
    return "down" if paddle_frect.pos[1] + paddle_frect.size[1] / 2 < ball_frect.pos[1] + ball_frect.size[1] / 2 else "up"