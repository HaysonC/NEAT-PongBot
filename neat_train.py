# neat_train.py
# NEAT Training script for the Pong game.
"""
This script trains a NEAT agent to play the Pong game against other NEAT agents and a chaser agent.

The script uses the NEAT-Python library to train the agent. The agent is evaluated based on its performance in matches

against other agents. The fitness function rewards the agent for hitting the ball and penalizes it for missing the ball or

not moving (at the start). The agent is trained for a predefined number of generations.

STRUCTURE:

- The script defines the game constants, fitness function constants, and evaluation constants.
- eval_genomes: The scripts create a pool of genomes and evaluates them in matches against other genomesfunction.
    - Each genome competes in a predefined number of matches against randomly selected opponent
    and once with a chaser agent.
- simulate match: The script simulates a single match between two genomes and updates their fitness scores with the below update_fitness function.
- update_fitness: The script updates the fitness score of the AI based on the game state and updates the game state.
- run: The script runs the NEAT algorithm with the given configuration.
- The script runs the NEAT algorithm for a predefined number of generations and saves the best genome to a file.
"""

import pickle
import time
from copy import deepcopy
from random import random
from typing import Callable
import neat
import numpy as np
from neat import Checkpointer
import os
from Analytics import Neat_Analytics
from chaser_ai import chaser_dummy_neat as chaser, chaser_dummy_neat
from dummy_neat import dummy_neat
from playPong import Game, visualize_game_loop

# CURRENT STATE OF THE ART OF THE LAST RUN:
# It is marginally better than human player when we don't focus, but if we lock in we can beat it fairly easily
# TODO: CURRENT ISSUES (in order of importance and ease of implementation):
# TODO: Implements Analytic information about the training:
#      - Number of generations
#      - Time taken for each generation and the total time
#      - Number of matches played
#      - Penalties and rewards given
#      - Ability to beat chaser
# TODO: Implement 2 more inputs: opponent's paddle's x, y;
#      - Restructure the neural network to take in 6 inputs
#      - Restructure the dummy classes
# TODO: Implement better genetic algorithm for the NEAT:
#      - Dissolve not moving features once it consistently moves and beat chaser
# TODO: When saving a checkpoint and winner, save the generation number and a date time
# TODO: Implement CPU or GPU usage
# TODO: Implement a app view for how and when the reward is given
# TODO: Implement NN visualization
# TODO: Restructure the whole code to accommodate other games (for example, use generation *args and **kwargs, and build subclasses for the pong game)

# Game constants
BALL_SIZE = 10
TABLE_WIDTH = 440
TABLE_HEIGHT = 280
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 70
MAX_SCORE = 5


# Fitness function constants
HIT_REWARD = 0.1
MISS_PENALTY = 1.2
OPPONENT_WIN_PENALTY = 1.5
PLAYER_WIN_REWARD = 0.2
NOT_MOVING_PENALTY = 0.00
WINRATE_NOT_PENALIZE = 0.3


# Evaluation Constants
MATCHES_PER_GENOME = 8
GENERATION = 1000
NUM_POP = 30
print("Estimated time for num pop: ", GENERATION * NUM_POP * MATCHES_PER_GENOME)

CONFIG_PATH = "config-fc.txt"
p = neat.Population(neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                       CONFIG_PATH))

def update_fitness(game: Game,
                   genome: neat.genome,
                   opponent_genome: neat.genome | Callable,
                   updateGame: bool = True ) -> bool:
    """
    Update the fitness score of the AI based on the game state.

    :param game: The game object representing the current state of the game.
    :param genome: The genome object representing the AI player on the left side.
    :param opponent_genome: The genome object representing the AI player on the right side.
    :param updateGame: Whether to update the game state or not, mostly for testing purposes if it is set to False
    :return: if the game should continue or not
    """
    if updateGame:
        res = game.update()
    else:
        temp = deepcopy(game)
        res =  temp.update()
        del temp

    def paddle_not_moving(g: Game, other_paddle: bool = False):
        if other_paddle:
            paddle = g.get_other_paddle()
        else:
            paddle = g.get_paddle()
        return paddle.get_move() == 0 or paddle.get_move() is None or \
            (paddle.get_move() == "up" and paddle.frect.pos[1] <= g.get_paddle().frect.size[1] + 0.1) or \
            (paddle.get_move() == "down" and paddle.frect.pos[1] >= g.get_table_size()[1] - paddle.frect.size[1] - 0.1)

    if (l := analytics.get_latest("winRate")) is None or l < WINRATE_NOT_PENALIZE:
        p =  NOT_MOVING_PENALTY * (1-l/WINRATE_NOT_PENALIZE)
        if paddle_not_moving(game):
            genome.fitness -= p
        if paddle_not_moving(game, other_paddle=True):
            opponent_genome.fitness -= p

    # Check for game over conditions
    if res  == -2:
        genome.fitness -= OPPONENT_WIN_PENALTY
        opponent_genome.fitness += PLAYER_WIN_REWARD
        if isinstance(genome, dummy_neat):
            analytics.add_win()
        return False  # End the game
    elif res  == 2:
        genome.fitness += PLAYER_WIN_REWARD
        opponent_genome.fitness -= OPPONENT_WIN_PENALTY
        if isinstance(opponent_genome, dummy_neat):
            analytics.add_win()
        return False  # End the game

    if res  == -1:
        genome.fitness -= MISS_PENALTY
    elif res  == 1:
        opponent_genome.fitness -= MISS_PENALTY

    if res  == 3:
        genome.fitness += HIT_REWARD
    elif res  == -3:
        opponent_genome.fitness += HIT_REWARD

    return True  # Continue the game


def simulate_match(genome: neat.DefaultGenome | dummy_neat,
                   opponent_genome: neat.DefaultGenome | dummy_neat,
                   config: neat.config.Config,
                   max_steps: int = 1500) -> None:
    """
    Simulate a single match between two genomes and update their fitness scores.

    :param genome: The participant genome, or a dummy_neat object if we use a different type of agent.
                   Note that if a dummy_neat object is passed, the opponent is not really trained by NEAT
    :param opponent_genome: The opponent genome.
    :param config: NEAT configuration object.
    :param max_steps: Maximum steps to prevent infinite games.
    """
    # Create neural networks for both genomes
    if isinstance(genome, dummy_neat):
        net = genome.net
    elif isinstance(genome, neat.DefaultGenome):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        raise ValueError("Invalid genome type")

    if isinstance(opponent_genome, dummy_neat):
        opponent_net = opponent_genome.net
    elif isinstance(opponent_genome, neat.DefaultGenome):
        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
    else:
        raise ValueError("Invalid genome type")

    # Initialize game
    game = Game(
        paddle_size=(PADDLE_WIDTH, PADDLE_HEIGHT),
        ball_size=(BALL_SIZE, BALL_SIZE),
        table_size=(TABLE_WIDTH, TABLE_HEIGHT)
    )

    should_continue = True

    steps = 0  # To prevent infinite games


    while should_continue and steps < max_steps:
        steps += 1

        ball = game.get_ball()
        player_paddle = game.get_paddle()
        opponent_paddle = game.get_other_paddle()


        player_output = net.activate((ball.frect.pos[0] - player_paddle.frect.pos[0],
                                       ball.frect.pos[1],
                                       player_paddle.frect.pos[0],
                                       player_paddle.frect.pos[1]
                                       ))

        opponent_output = opponent_net.activate((opponent_paddle.frect.pos[0] - ball.frect.pos[0],
                                                 ball.frect.pos[1],
                                                 opponent_paddle.frect.pos[0],
                                                 opponent_paddle.frect.pos[1]
                                                 ))

        player_move = np.argmax(player_output) - 1  # Outputs: -1, 0, 1
        player_paddle.set_move(player_move)
        opponent_move = np.argmax(opponent_output) - 1  # Outputs: -1, 0, 1
        opponent_paddle.set_move(opponent_move)

        # Update game state and fitness
        should_continue = update_fitness(game, genome, opponent_genome)




def eval_genomes(genomes: list[tuple[int, neat.genome]], config: neat.config.Config) -> None:
    """
    Evaluate the fitness of each genome in the population by having them compete against other genomes.

    Each genome competes in a predefined number of matches against randomly selected opponents.
    Fitness scores are updated based on performance in these matches.

    :param genomes: List of tuples (genome_id, genome) representing the population.
    :param config: Configuration object for the NEAT algorithm.
    """
    # create a list of all genomes and add some chasers to it
    print(genomes)
    n = len(genomes)
    matchPlayed = [0 for _ in range(n)]
    for genome_id, genome in genomes:
        genome.fitness = 0
    # create a shallow copy of the list of genomes
    # Round-robin tournament
    for i, (genome_id, genome) in enumerate(genomes):
        for _ in range(MATCHES_PER_GENOME - 1):
            choice = int(random() * n)
            if choice == i:
                choice = (choice + 1) % n
            matchPlayed[i] += 1
            matchPlayed[choice] += 1
            opponent_genome_id, opponent_genome = genomes[choice]
            simulate_match(genome, opponent_genome, config)
            simulate_match(opponent_genome, genome, config)

        simulate_match(genome, chaserbot, config)
        simulate_match(chaserbot, genome, config)
        matchPlayed[i] += 1
    for i,j in enumerate(matchPlayed):
        genomes[i][1].fitness /= j

    # remove all dummy agents from the list of genomes
    genomes = [(genome_id, genome) for genome_id, genome in genomes if not isinstance(genome, dummy_neat)]
    best = max(genomes, key=lambda x: x[1].fitness)
    with open("models/train_best.pkl", "wb") as f:
        pickle.dump(best[1], f)

    # update the analytics
    analytics.append({
        "winRate": analytics.get_winRate(2, n),
        "bestGenome": best[1],
        "bestGenomesFitness": best[1].fitness,
        "bestGenomesSpecies": p.species.get_species_id(best[1].key),
        "avgFitness": sum([g.fitness for _, g in genomes]) / len(genomes),
    })


def run(config_path: str, show: bool = False, useLastRun = False) -> None:
    global p
    """
    Run the NEAT algorithm with the given configuration.

    :param config_path: Path to the configuration file for the NEAT algorithm.
    :param show: Whether to show the visualization of the winning genome and a sample game.
    :param useLastRun: Whether to use the last run to continue the training
    :param generation: Number of generation to train
    """

    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run
    # Create the population
    p = neat.Population(config)
    if useLastRun:
        # find the latest checkpoint
        path = "models"
        files = os.listdir(path)
        files = [f for f in files if f.startswith("lastcheckpoint")]
        if len(files) > 0:
            files.sort()
            p = Checkpointer().restore_checkpoint("models/" + files[-1])
        else:
            print(f"Last checkpoint not found in directory models, starting new run or abort and check the directory)")
            a = input("Abort? (Y/N)")
            if a == "Y":
                exit()
    # Add reporters to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT for up to 50 generations
    winner = p.run(eval_genomes, n=GENERATION)
    # how do you save last run?
    if useLastRun:
        Checkpointer().save_checkpoint(p.config, p.population, p.species, GENERATION)
    if useLastRun:
        path = "models/lastcheckpoint" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(p, f)

    print(f"Best genome -> {winner}")

    # Save the best genome for later use
    with open("models/best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    if show:


        # Visualize the game loop with the winner vs a Human Player
        game = Game()
        import neat_inference  # Ensure this module is properly set up
        from playPong import HumanPlayer

        # Load the winner's neural network
        neat_inference.model = neat.nn.FeedForwardNetwork.create(winner, config)

        analytics.plot("winRate")
        analytics.plot("time")
        analytics.plot("timesForEachGen")

        analytics.save_to_file("analytics.json")

        # Start the visualization game loop
        visualize_game_loop(game, player1=neat_inference.pong_ai, player2=HumanPlayer(), tickTime=0.01)

        # Visualize the winning genome as a graph
        # analytics.visualize_genome(winner)




analytics = Neat_Analytics(NUM_POP)

def main():

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-fc.txt")
    run(config_path, show=True)


if __name__ == "__main__":
    import pygame

    chaserbot = chaser_dummy_neat(paddle_size=(PADDLE_WIDTH, PADDLE_HEIGHT),
                                  ball_size=(BALL_SIZE, BALL_SIZE),
                                  table_size=(TABLE_WIDTH, TABLE_HEIGHT))
    pygame.init()
    clock = pygame.time.Clock()
    main()

