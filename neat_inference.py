import neat, pickle, os
import numpy as np

from PongAIvAi import fRect, init_game
local_dir = os.path.dirname(__file__)
# current
MODEL_PATH = "models/first_genome_worked.pkl"
MODEL_PATH2 = "models/train_best.pkl"
config_path = os.path.join(local_dir, "config-fc.txt")

def load_model(config_path: str =os.path.join(local_dir, "config-fc.txt"), model_path:str = MODEL_PATH) -> neat.nn.FeedForwardNetwork:
    """
    Load the genome from the file and create a neural network from it.

    :param config_path: The path to the configuration file
    :param model_path: The path to the model file
    :return: The neural network created from the genome
    """

    # Load the configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the best genome
    with open(model_path, "rb") as f:
        best_genome = pickle.load(f)

    # Create a neural network from the loaded genome
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    return net


model = load_model(config_path, MODEL_PATH)
model2 = load_model(config_path, MODEL_PATH2)


def pong_ai(paddle_frect: fRect,
            other_paddle_frect: fRect,
            ball_frect: fRect,
            table_size: tuple,
            mod = 1) -> str | None:
    """
    Main trigger for the official Pong Game

    :param paddle_frect: The paddle object
    :param other_paddle_frect: The other paddle object
    :param ball_frect: The ball object
    :param table_size: The size of the table
    :param mod: The model to use, either 1 or 2

    :return: The final output of the AI, either "up", "down", or None
    """
    if mod == 1:
        output = model.activate((abs(ball_frect.pos[0] - paddle_frect.pos[0])
                                 , ball_frect.pos[1],
                          paddle_frect.pos[0], paddle_frect.pos[1],
                            ))
    else:
        output = model2.activate((abs(ball_frect.pos[0] - paddle_frect.pos[0])
                                  ,ball_frect.pos[1],
                          paddle_frect.pos[0], paddle_frect.pos[1],
                            ))
    op = np.argmax(output) - 1
    return None if op == 0 else "up" if op == 1 else "down"


def pong_ai2(paddle_frect: fRect,
            other_paddle_frect: fRect,
            ball_frect: fRect,
            table_size: tuple) -> str | None:
    """
    Main trigger for the official Pong Game

    :param paddle_frect: The paddle object
    :param other_paddle_frect: The other paddle object
    :param ball_frect: The ball object
    :param table_size: The size of the table

    :return: The final output of the AI, either "up", "down", or None
    """
    return pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size, 2)

def main():
    pass


if __name__ == '__main__':
    main()
    init_game(player1=pong_ai)