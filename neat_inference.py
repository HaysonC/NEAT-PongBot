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

def make_inference(net, inputs):
    # Use the neural network to make predictions
    output = net.activate(inputs)
    return output

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-fc.txt")

    # Load the best genome and create the neural network for inference
    net = load_model(config_path)

    # Example input to the neural network (you should replace this with actual input values)
    # inputs = (ball_dx, ball_dy, platform_y_distance, ball_y_distance)
    inputs = (0.5, -0.2, 10, 20)  # Dummy values, adjust as necessary

    # Make an inference
    output = make_inference(net, inputs)
    finalOP = output.index(max(output)) - 1
    print("Inference output:", finalOP)