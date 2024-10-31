import neat, os
import pickle
from graphviz import Digraph

# Remove this import and import the real game
from sim import *

# Fitness function
HIT_REWARD = 2
MISS_PENALTY = 5
OPPONENT_WIN_PENALTY = 10
PLAYER_WIN_REWARD = 20

def update_fitness(ball, player_paddle, g):
    """
    Update the fitness score of the AI based on the game state.
    Parameters:
    ball (Ball): The ball object containing its current position and scores.
    player_paddle (Paddle): The player's paddle object containing its current position.
    g (Genome): The genome object representing the AI's neural network.
    Returns:
    bool: False if the game should end, True if the game should continue.
    """

    # Check for game over conditions
    if ball.score_opponent >= MAX_SCORE:
        g.fitness -= OPPONENT_WIN_PENALTY
        return False  # End the game
    elif ball.score_player >= MAX_SCORE:
        g.fitness += PLAYER_WIN_REWARD
        return False  # End the game

    if abs(ball.x) == abs(player_paddle.x - BALL_RADIUS):
        if player_paddle.y - 1/2 * PADDLE_HEIGHT <= ball.y <= player_paddle.y + 1/2 * PADDLE_HEIGHT:
            # Reward for successful hit
            g.fitness += HIT_REWARD
        else:
            # Penalty for missing the ball
            g.fitness -= MISS_PENALTY
            return False  # End the game
        
    return True  # Continue the game

def eval_genomes(genomes, config):
    """
    Evaluate the fitness of each genome in the population.
    Args:
        genomes (list): List of tuples (genome_id, genome) representing the population.
        config (neat.Config): Configuration object for the NEAT algorithm.
    The function initializes a neural network for each genome and simulates a game of Pong.
    The fitness of each genome is determined based on its performance in the game.
    """

    for _, g in genomes:
        # Create a neural network for the genome
        net = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 0  # Initialize the fitness score

        # pygame.init()
        # win = pygame.display.set_mode((WIDTH, HEIGHT))
        # pygame.display.set_caption("Pong Game")
        # clock = pygame.time.Clock()
        # font = pygame.font.SysFont("comicsans", 30)

        # Initialize game objects
        player_paddle = Paddle(30, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        opponent_paddle = Paddle(WIDTH - 30 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        ball = Ball()

        run = True
        while run:

            # clock.tick(FPS)
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         run = False

            # NOTE: modify to interact with real game
            # Opponent AI movement
            opponent_move = opponent_ai(ball, opponent_paddle)
            opponent_paddle.move(opponent_move)

            # Draw everything
            # draw_window(win, player_paddle, opponent_paddle, ball, font)

            # Input to the neural network: ball's movement and relative position to platform
            output = net.activate((
                ball.x, ball.y,
                player_paddle.x, player_paddle.y
            ))

            # Determine movement from network output (-1, 0, or 1)
            finalOP = output.index(max(output)) - 1
            
            # NOTE: Modify to pass decision back to game
            # Update platform position
            player_paddle.move(finalOP)

            # Move ball
            ball.move(player_paddle, opponent_paddle)

            # Incremental reward for staying in play
            g.fitness += 0.1

            run = update_fitness(ball, player_paddle, g)

        # pygame.quit()

def run(config_path):
    """
    Run the NEAT algorithm with the given configuration.

    Args:
        config_path (str): Path to the NEAT configuration file.

    Returns:
        None

    This function sets up the NEAT population, adds reporters for logging and statistics,
    runs the NEAT algorithm to evolve the population, and saves the best genome to a file.
    It also visualizes the winning genome.
    """

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes)

    print("Best fitness -> {}".format(winner))

    # Save the best genome for later use
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    # Visualize the winning genome
    visualize_genome(winner)

# TODO: Fix this function! Bugging!
def visualize_genome(genome):
    """Function to visualize the winning genome using Graphviz."""

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

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-fc.txt")
    run(config_path)