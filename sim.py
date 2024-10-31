import neat, os, pygame
import pickle

from neat_ai import *

# Colors and setup
RED = (250, 70, 80)
WHITE = (255, 255, 255)
WIDTH = 780
HEIGHT = 480
BALL_RADIUS = 15
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 70
PADDLE_SPEED = 1
FPS = 60
MAX_SCORE = 5

def load_winner_genome(file_path):
    with open(file_path, "rb") as f:
        winner_genome = pickle.load(f)
    return winner_genome

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong AI Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("comicsans", 30)

    # Load the trained genome
    winner_genome = load_winner_genome("best_genome.pkl")
    config_path = os.path.join(os.path.dirname(__file__), "config-fc.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    # Initialize game objects
    player_paddle = Paddle(30, HEIGHT // 2 - PADDLE_HEIGHT // 2)
    opponent_paddle = Paddle(WIDTH - 30 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)
    ball = Ball()

    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Opponent AI movement
        opponent_move = opponent_ai(ball, opponent_paddle)
        opponent_paddle.move(opponent_move)

        output = net.activate((ball.x, ball.y, player_paddle.x, player_paddle.y))
        finalOP = output.index(max(output)) - 1
        player_paddle.move(finalOP)

        # Move ball
        ball.move(player_paddle, opponent_paddle)

        # Draw everything
        draw_window(win, player_paddle, opponent_paddle, ball, font)

        if ball.score_opponent >= MAX_SCORE or ball.score_player >= MAX_SCORE:
            run = False  # End game when max score is reached

    pygame.quit()

if __name__ == "__main__":
    main()