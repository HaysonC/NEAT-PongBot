import pickle
import time
from copy import deepcopy
from random import random
from typing import Callable
import neat
from neat import Checkpointer
import os
from graphviz import Digraph
from chaser_ai import chaser_dummy_neat as chaser
from dummy_neat import dummy_neat
from playPong import Game, visualize_game_loop

# Game constants
BALL_SIZE = 10

TABLE_WIDTH = 440
TABLE_HEIGHT = 280

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 70

MAX_SCORE = 5



# Fitness function constants
HIT_REWARD = 0.5
MISS_PENALTY = 0.7
OPPONENT_WIN_PENALTY = 0
PLAYER_WIN_REWARD = 0
NOT_MOVING_PENALTY = 0.07


# Evaluation Constants
MATCHES_PER_GENOME = 6
AMOUNT_OF_CHASERS = 5
GENERATION = 20

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

    def paddle_not_moving(g: Game, other_paddle = False):
        if not other_paddle:
            return g.get_paddle().get_move() == 0 or g.get_paddle().get_move() is None or \
            (g.get_paddle().get_move() == "up" and g.get_paddle().frect.pos[1] <= g.get_paddle().frect.size[1] + 0.1) or \
            (g.get_paddle().get_move() == "down" and g.get_paddle().frect.pos[1] >= g.get_table_size()[1] - g.get_paddle().frect.size[1] - 0.1)
        else:
            return g.get_other_paddle().get_move() == 0 or g.get_other_paddle().get_move() is None or \
            (g.get_other_paddle().get_move() == "up" and g.get_other_paddle().frect.pos[1] <= g.get_other_paddle().frect.size[1] + 0.1) or \
            (g.get_other_paddle().get_move() == "down" and g.get_other_paddle().frect.pos[1] >= g.get_table_size()[1] - g.get_other_paddle().frect.size[1] - 0.1)

    if paddle_not_moving(game):
        genome.fitness -= NOT_MOVING_PENALTY
    if paddle_not_moving(game, other_paddle=True):
        opponent_genome.fitness -= NOT_MOVING_PENALTY

    # Check for game over conditions
    if res  == -2:
        genome.fitness -= OPPONENT_WIN_PENALTY
        opponent_genome.fitness += PLAYER_WIN_REWARD
        return False  # End the game
    elif res  == 2:
        genome.fitness += PLAYER_WIN_REWARD
        opponent_genome.fitness -= OPPONENT_WIN_PENALTY
        return False  # End the game

    if res  == -1:
        genome.fitness -= MISS_PENALTY
    elif res  == 1:
        opponent_genome.fitness += MISS_PENALTY

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
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    if isinstance(opponent_genome, neat.DefaultGenome):
        opponent_net = neat.nn.FeedForwardNetwork.create(opponent_genome, config)
    elif isinstance(opponent_genome, dummy_neat):
        opponent_net = opponent_genome.net
    else:
        raise ValueError("Invalid opponent genome type")
    # Initialize game
    game = Game(
        paddle_size=(PADDLE_WIDTH, PADDLE_HEIGHT),
        ball_size=(BALL_SIZE, BALL_SIZE),
        table_size=(TABLE_WIDTH, TABLE_HEIGHT)
    )
    player_paddle = game.get_paddle()
    opponent_paddle = game.get_other_paddle()
    ball = game.get_ball()
    should_continue = True

    steps = 0  # To prevent infinite games

    while should_continue and steps < max_steps:
        steps += 1


        player_output = net.activate((ball.frect.pos[0],
                                       ball.frect.pos[1],
                                       player_paddle.frect.pos[0],
                                       player_paddle.frect.pos[1]
                                       ))

        opponent_output = opponent_net.activate((ball.frect.pos[0],
                                                 ball.frect.pos[1],
                                                 opponent_paddle.frect.pos[0],
                                                 opponent_paddle.frect.pos[1]
                                                 ))

        player_move = player_output.index(max(player_output)) - 1  # Outputs: -1, 0, 1
        player_paddle.set_move(player_move)
        opponent_move = opponent_output.index(max(opponent_output)) - 1  # Outputs: -1, 0, 1
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
    global GENERATION
    # create a list of all genomes and add some chasers to it
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
        simulate_match(genome, chaser(), config)
    for i,j in enumerate(matchPlayed):
        genomes[i][1].fitness /= j
    GENERATION -= 1

    # remove all dummy agents from the list of genomes
    genomes = [(genome_id, genome) for genome_id, genome in genomes if not isinstance(genome, dummy_neat)]


def run(config_path: str, show: bool = False, useLastRun = False, generation:int = GENERATION) -> None:
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
    winner = p.run(eval_genomes, n=generation)
    # how do you save last run?
    if useLastRun:
        Checkpointer().save_checkpoint(p.config, p.population, p.species, generation)
    # TODO: when saving a checkpoint and winner, save the generation number and a date time
    # save it to lastcheckpoint.pkl, dont replace anyfile
    # format date time to yyyy-mm-dd-hh-mm-ss
    if useLastRun:
        path = "models/lastcheckpoint" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(p, f)

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



# TODO: Implement a app view for how and when the reward is given
# TODO: Implement CPU or GPU usage
# TODO: when saving a checkpoint and winner, save the generation number and a date time
# TODO: Implement NN visualization