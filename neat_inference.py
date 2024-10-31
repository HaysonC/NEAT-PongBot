import neat, pickle, os

def load_model(config_path):

    # Load the configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the best genome
    with open("best_genome.pkl", "rb") as f:
        best_genome = pickle.load(f)

    # Create a neural network from the loaded genome
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    return net

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):

    '''
    Main trigger for the official Pong Game
    '''
    output = model.activate((ball_frect.pos[0], ball_frect.pos[1],
                          paddle_frect.pos[0], paddle_frect.pos[1]))
    
    op = output.index(max(output)) - 1
    if op == 0:
        finalOP = None
    elif op == 1:
        finalOP = "up"
    else:
        finalOP = "down"

    return finalOP

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config-fc.txt")
model = load_model(config_path)