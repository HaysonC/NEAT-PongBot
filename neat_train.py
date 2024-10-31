import pickle

import neat, os
from graphviz import Digraph

from playPong import Game, visualize_game_loop

# opponent AI would be a simple chaser for now
from chaser_ai import pong_ai as opponent_ai

# Fitness function
HIT_REWARD = 2
MISS_PENALTY = 5
OPPONENT_WIN_PENALTY = 10
PLAYER_WIN_REWARD = 20
MAX_SCORE = 5


def update_fitness(game: Game, genome: neat.genome, side: 1 | -1 = 1) -> bool:
    """
    Update the fitness score of the AI based on the game state.

    Game state would be updated based on the move of the AI which shall be previously determined by the neural network.
    and told to the game object via Game.set_move() method.

    :param game: The game object representing the current state of the game.
    :param genome: The genome object representing the AI player.
    :param side: The side of the AI player (1 for left, -1 for right).
    :return: False if the game should end, True if the game should continue.
    """
    res = game.update()

    # Check for game over conditions
    if res*side == -2:
        genome.fitness -= OPPONENT_WIN_PENALTY
        return False  # End the game
    elif res*side == 2:
        genome.fitness += PLAYER_WIN_REWARD
        return False  # End the game
    if res*side == -1:
        genome.fitness -= MISS_PENALTY
    elif res*side == 1:
        genome.fitness += HIT_REWARD

    return True  # Continue the game

def eval_genomes(genomes: list[tuple[int, neat.genome]], config: neat.config.Config) -> None:
    """
    Evaluate the fitness of each genome in the population.

    The function initializes a neural network for each genome and simulates a game of Pong.
    The fitness of each genome is determined based on its performance in the game.

    :param genomes: List of tuples (genome_id, genome) representing the population.
    :param config: Configuration object for the NEAT algorithm.
    """

    for _, g in genomes:
        # Create a neural network for the genome
        net = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 0  # Initialize the fitness score

        # Initialize game objects
        game = Game()
        player_paddle = game.get_paddle()
        opponent_paddle = game.get_other_paddle()
        ball = game.get_ball()
        run = True

        while run:

            opponent_move = opponent_ai(opponent_paddle.frect, player_paddle.frect, ball.frect, game.get_table_size())
            opponent_paddle.set_move(opponent_move)

            output = net.activate((
                ball.frect.pos[0], ball.frect.pos[1],
                player_paddle.frect.pos[0], player_paddle.frect.pos[1]
            ))

            # Determine movement from network output (-1, 0, or 1)
            finalOP = output.index(max(output)) - 1
            
            # NOTE: Modify to pass decision back to game
            # Update platform position
            player_paddle.set_move(finalOP)


            # Incremental reward for staying in play
            g.fitness += 0.1

            run = update_fitness(game, g, side = 1)

        # pygame.quit()

def run(config_path: str, show = False) -> None:
    """
    Run the NEAT algorithm with the given configuration.

    This function sets up the NEAT population, adds reporters for logging and statistics,
    runs the NEAT algorithm to evolve the population, and saves the best genome to a file.
    It also visualizes the winning genome.

    :param config_path: Path to the configuration file for the NEAT algorithm.
    :param show: Whether to show the visualization of the winning genome and a sample game.
    """

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes)

    print(f"Best genome -> {winner}")

    # Save the best genome for later use
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    if show:
        """ 
        # Visualize the winning genome as a graph
        visualize_genome(winner)
        """
        # Visualize the game loop
        game = Game()
        import neat_inference
        from playPong import HumanPlayer
        # load the winner to the neat_inference
        neat_inference.model = neat.nn.FeedForwardNetwork.create(winner, config)
        # break down the iterable
        visualize_game_loop(game, player1=neat_inference.pong_ai, player2=HumanPlayer())




# TODO: Fix this function! Bugging!
def visualize_genome(genome) -> Digraph:
    """
    Function to visualize the winning genome using Graphviz.

    :param genome: The winning genome object.
    :return: The Digraph object representing the neural network.
    """

    dot = Digraph(comment='Neural Network', format='png')

    # Define node names for input and output
    node_names = {
        -1: 'ball.x', -2: 'ball.y', -3: 'paddle.x', -4: 'paddle.y',  # Input nodes
        0: 'Output_U', 1: 'Output_D', 2: 'Output_Stay'  # Output nodes
    }

    # Define different colors for input, hidden, and output nodes
    for node in genome.nodes:
        if node in node_names:
            if node < 0:
                dot.node(str(node), label=node_names[node], style='filled', fillcolor='lightblue')  # Input
            else:
                dot.node(str(node), label=node_names[node], style='filled', fillcolor='lightgreen')  # Output
        else:
            dot.node(str(node), label=f'Hidden {node}', style='filled', fillcolor='lightgrey')  # Hidden

    # Add edges with weights
    for conn in genome.connections.values():
        if conn.enabled:
            dot.edge(str(conn.key[0]), str(conn.key[1]), label=f"{conn.weight:.2f}", color='grey')

    # Render the graph and view it
    dot.render('neural_network', view=True, cleanup=True)

    print("Visualization generated: neural_network.png")
    return dot

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-fc.txt")
    run(config_path, show=True)