import neat, os, pygame, random
import pickle

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

class Ball:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 3
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed
        self.score_player = 0
        self.score_opponent = 0

    def draw(self, win):
        pygame.draw.circle(win, RED, (self.x, self.y), BALL_RADIUS)

    def move(self, player_paddle, opponent_paddle):
        # Ball movement
        self.x += self.dx
        self.y += self.dy

        # Gradually increase speed
        self.speed += 0.05
        self.dx = self.dx / abs(self.dx) * self.speed
        self.dy = self.dy / abs(self.dy) * self.speed

        # Bounce off top or bottom wall
        if self.y - BALL_RADIUS <= 0 or self.y + BALL_RADIUS >= HEIGHT:
            self.dy *= -1

        # Check collision with player paddle
        if (self.dx < 0 and player_paddle.x < self.x < player_paddle.x + PADDLE_WIDTH and
            player_paddle.y < self.y < player_paddle.y + PADDLE_HEIGHT):
            self.dx *= -1

        # Check collision with opponent paddle
        elif (self.dx > 0 and opponent_paddle.x < self.x < opponent_paddle.x + PADDLE_WIDTH and
              opponent_paddle.y < self.y < opponent_paddle.y + PADDLE_HEIGHT):
            self.dx *= -1

        # Check if the ball goes past the paddles (scoring)
        if self.x < 0:
            self.score_opponent += 1
            self.reset()
        elif self.x > WIDTH:
            self.score_player += 1
            self.reset()

    def reset(self):
        # Reset ball position and speed
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 3
        self.dx = random.choice([-1, 1]) * self.speed
        self.dy = random.choice([-1, 1]) * self.speed

class Paddle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vel = PADDLE_SPEED

    def draw(self, win):
        pygame.draw.rect(win, WHITE, (self.x, self.y, PADDLE_WIDTH, PADDLE_HEIGHT))

    def move(self, direction):
        # Move paddle up or down within screen bounds
        if direction == -1 and self.y > 0:
            self.y -= self.vel
        elif direction == 1 and self.y < HEIGHT - PADDLE_HEIGHT:
            self.y += self.vel

def opponent_ai(ball, opponent_paddle):
    # Simple AI to track the ball's y position
    if ball.y < opponent_paddle.y + PADDLE_HEIGHT // 2:
        return -1
    elif ball.y > opponent_paddle.y + PADDLE_HEIGHT // 2:
        return 1
    return 0

def draw_window(win, player_paddle, opponent_paddle, ball, font):
    win.fill((0, 0, 0))  # Solid black background
    player_paddle.draw(win)
    opponent_paddle.draw(win)
    ball.draw(win)

    # Display scores
    player_score_text = font.render(f"Player: {ball.score_player}", True, WHITE)
    opponent_score_text = font.render(f"Opponent: {ball.score_opponent}", True, WHITE)
    win.blit(player_score_text, (WIDTH // 4, 20))
    win.blit(opponent_score_text, (3 * WIDTH // 4, 20))

    pygame.display.update()

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
    winner_genome = load_winner_genome("7NovGenome.pkl")
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