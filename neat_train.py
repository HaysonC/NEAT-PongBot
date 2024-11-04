import pickle
from typing import Callable

import neat
import os
from graphviz import Digraph
from neat import Checkpointer

import chaser_ai
from playPong import Game, visualize_game_loop
import random

# Fitness function constants
HIT_REWARD = 0.5
MISS_PENALTY = 0.7
OPPONENT_WIN_PENALTY = 0
PLAYER_WIN_REWARD = 0
MAX_SCORE = 5
MATCHES_PER_GENOME = 3  # Number of opponents each genome will face


def update_fitness(game: Game,
                   genome: neat.genome,
                   side: int = 1) -> bool:
    """
    Update the fitness score of the AI based on the game state.

    :param game: The game object representing the current state of the game.
    :param genome: The genome object representing the AI player.
    :param side: The side of the AI player (1 for left, -1 for right).
    :return: False if the game should end, True if the game should continue.
    """
    res = game.update()
    if not isinstance(genome,neat.DefaultGenome):
        return res == 0
    # Check for game over conditions
    if res * side == -2:
        genome.fitness -= OPPONENT_WIN_PENALTY
        return False  # End the game
    elif res * side == 2:
        genome.fitness += PLAYER_WIN_REWARD
        return False  # End the game

    if res * side == -1:
        genome.fitness -= MISS_PENALTY

    if res * side == 3:
        genome.fitness += HIT_REWARD

    return True  # Continue the game


def simulate_match(genome: neat.genome,
                   opponent_genome: neat.genome | Callable
                   , config, table_size=(440, 280), score_to_win=5, max_steps=1000):
    """
    Simulate a single match between two genomes and update their fitness scores.

    :param genome: The participant genome.
    :param opponent_genome: The opponent genome.
    :param config: NEAT configuration object.
    :param table_size: Size of the game table.
    :param score_to_win: Points needed to win the game.
    :param max_steps: Maximum steps to prevent infinite games.
    """
    # Create neural networks for both genomes
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    flag = True
    if isinstance(opponent_genome, neat.DefaultGenome):
        flag = False
        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)


    # Initialize game
    game = Game()
    player_paddle = game.get_paddle()
    opponent_paddle = game.get_other_paddle()
    ball = game.get_ball()
    run = True

    steps = 0  # To prevent infinite games

    while run and steps < max_steps:
        steps += 1

        # Genome's move
        player_output = net.activate((
            ball.frect.pos[0], ball.frect.pos[1],
            player_paddle.frect.pos[0], player_paddle.frect.pos[1]
        ))
        player_move = player_output.index(max(player_output)) - 1  # Outputs: -1, 0, 1
        player_paddle.set_move(player_move)
        if not flag:
            # Opponent genome's move
            opponent_output = opponent_net.activate((
                ball.frect.pos[0], ball.frect.pos[1],
                opponent_paddle.frect.pos[0], opponent_paddle.frect.pos[1]
            ))
        else:
            opponent_output = opponent_genome(*game.get_game_state())
        opponent_move = opponent_output.index(max(opponent_output)) - 1  # Outputs: -1, 0, 1
        opponent_paddle.set_move(opponent_move)

        # Update game state and fitness
        run_player = update_fitness(game, genome, side=1)
        run_opponent = update_fitness(game, opponent_genome, side=-1)
        run = run_player and run_opponent


def eval_genomes(genomes: list[tuple[int, neat.genome]], config: neat.config.Config) -> None:
    """
    Evaluate the fitness of each genome in the population by having them compete against other genomes.

    Each genome competes in a predefined number of matches against randomly selected opponents.
    Fitness scores are updated based on performance in these matches.

    :param genomes: List of tuples (genome_id, genome) representing the population.
    :param config: Configuration object for the NEAT algorithm.
    """
    # Convert genomes to a list for easy access
    genome_list = list(genomes)
    num_genomes = len(genome_list)
    for genome_id, genome in genome_list:
        genome.fitness = 0

    for i, (genome_id, genome) in enumerate(genome_list):

        # Determine opponents for this genome

        opponents = random.sample(genome_list[:i] + genome_list[i + 1:],
                                  min(MATCHES_PER_GENOME, num_genomes - 1))


        for opponent_id, opponent_genome in opponents:
            # Simulate match between genome and opponent_genome
            simulate_match(genome, opponent_genome, config, score_to_win=MAX_SCORE)

        for i in range(len(opponents)//2):
            simulate_match(genome, chaser_ai.pong_ai, config, score_to_win=MAX_SCORE)

        # Optionally, average the fitness over multiple matches
        genome.fitness /= MATCHES_PER_GENOME


def run(config_path: str, show: bool = False, useLastRun = True, generation:int = 60) -> None:
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
    # Add reporters to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT for up to 50 generations
    winner = p.run(eval_genomes, n=generation)

    print(f"Best genome -> {winner}")

    # Save the best genome for later use
    with open("models/best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    if show:
        # Visualize the winning genome as a graph
        # visualize_genome(winner)

        # Visualize the game loop with the winner vs a Human Player
        game = Game()
        import neat_inference  # Ensure this module is properly set up
        from playPong import HumanPlayer

        # Load the winner's neural network
        neat_inference.model = neat.nn.FeedForwardNetwork.create(winner, config)

        # Start the visualization game loop
        visualize_game_loop(game, player1=neat_inference.pong_ai, player2=HumanPlayer())


# TODO: Fix this function! Bugging!
def visualize_genome(genome: neat.genome) -> Digraph:
    """
    Function to visualize the winning genome using Graphviz.

    :param genome: The winning genome object.
    :return: The Digraph object representing the neural network.
    """

    dot = Digraph(comment='Neural Network', format='png')
    dot.attr(rankdir='LR')  # Left to right layout

    # Define node names for input and output
    node_names = {
        -1: 'ball.x', -2: 'ball.y', -3: 'paddle.x', -4: 'paddle.y',  # Input nodes
        0: 'Output_U', 1: 'Output_D', 2: 'Output_Stay'  # Output nodes
    }

    # Identify all nodes
    input_nodes = [-1, -2, -3, -4]
    output_nodes = [0, 1, 2]
    hidden_nodes = [node for node in genome.nodes if node not in input_nodes and node not in output_nodes]

    # Add input nodes
    for node in input_nodes:
        if node in node_names:
            dot.node(str(node), label=node_names[node], style='filled', fillcolor='lightblue')

    # Add hidden nodes
    for node in hidden_nodes:
        dot.node(str(node), label=f'Hidden {node}', style='filled', fillcolor='lightgrey')

    # Add output nodes
    for node in output_nodes:
        if node in node_names:
            dot.node(str(node), label=node_names[node], style='filled', fillcolor='lightgreen')

    # Add edges with weights
    for conn in genome.connections.values():
        if conn.enabled:
            input_node = conn.key[0]
            output_node = conn.key[1]
            weight = conn.weight
            # Adjust the edge color based on weight direction
            if weight > 0:
                color = 'green'
            elif weight < 0:
                color = 'red'
            else:
                color = 'grey'
            dot.edge(str(input_node), str(output_node), label=f"{weight:.2f}", color=color)

    # Render the graph and view it
    dot.render('neural_network', view=True, cleanup=True)

    print("Visualization generated: neural_network.png")
    return dot


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-fc.txt")
    run(config_path, show=True)
