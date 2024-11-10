import random, math
from collections import deque, namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
# import gymnasium as gym
# import ale_py

from playPong import Game, visualize_game_loop

# opponent AI would be a simple chaser for now
# from chaser_ai import chaser_ai as opponent_ai

# Fitness function
PLAYER_WIN_REWARD = 5
OPPONENT_WIN_PENALTY = 5
HIT_REWARD = 1
MISS_PENALTY = 1
# NOT_MOVING_PENALTY = 0.05

# Hyperparameters
LEARNING_RATE = 1e-4

# Memory hyperparameters
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64  # Number of transitions sampled for training
STEPS_TO_UPDATE = 10

# Training hyperparameters
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 200
GAMMA = 0.999  # Discount factor
TAU = 0.05   # Soft update parameter

SEED = 42

TARGET_UPDATE_FREQ = 500 # this is in terms of time steps
OPP_UPDATE_FREQ = 5000 # time steps

torch.manual_seed(SEED)
random.seed(SEED)

# ======================Replay Buffer=======================

class ReplayMemory(object):
    def __init__(self, capacity):
        self.Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
        self.memory = deque([], maxlen=capacity)
    

    def push(self, *args):
        self.memory.append(self.Transition(*args))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):

    # Input parameters: ball.x, ball.y, paddle.x, paddle.y

    def __init__(self, num_actions=3):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent():

    def __init__(self, device):
        
        self.num_actions = 3

        self.policy_net = QNetwork(self.num_actions).to(device)
        self.target_net = QNetwork(self.num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.device = device
        self.policy_net = self.policy_net
        self.target_net = self.target_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.steps_done = 0
        self.num_steps_to_update = STEPS_TO_UPDATE

        self.Transitions = namedtuple("Transition", ("state", "action", "next_state", "reward"))

        self.reward_mean = 0
        self.reward_std = 1
    

    def select_action(self, state):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.inference_mode():
                action_values = self.policy_net(state.unsqueeze(0))
                action = action_values.max(1)[1] - 1 # select latest, remap to -1, 0, 1
                return torch.tensor([action[0]], device=self.device, dtype=torch.int32) # only most recent action
        else:
            return torch.tensor([random.randint(-1, 1)], device=self.device, dtype=torch.int32)
    

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE or self.steps_done % self.num_steps_to_update != 0:
            return
        
        # Sample a batch of transitions from the replay memory
        transitions = self.memory.sample(BATCH_SIZE)
        batch = self.Transitions(*zip(*transitions))

        # Extract the individual elements from the batch
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        if any(non_final_mask):  # Check if there are any non-final states
            non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        else:
            non_final_next_states = torch.tensor([], device=self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(state_batch) # gather() is too weak, let's select this ourselves
        state_action_values = q_values[range(q_values.size(0)), action_batch.long().squeeze()] 

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        if non_final_next_states.size(0) > 0:  # Ensure non_final_next_states is not empty
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        return state_action_values.mean().item()


    def get_reward(self, game):
        res = game.update()
        r = 0

        '''
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
            r -= NOT_MOVING_PENALTY
        '''

        # Check for game over conditions
        if res  == -2:
            r -= OPPONENT_WIN_PENALTY
            return (False, r)  # End the game
        
        if res == 2: 
            r += PLAYER_WIN_REWARD
            return (False, r)

        if res  == -1:
            r -= MISS_PENALTY

        if res  == 3:
            r += HIT_REWARD

        return (True, r)  # Continue the game
    
    def normalize_reward(self, reward, alpha=0.001):
        # Update running mean and standard deviation
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward
        self.reward_std = (1 - alpha) * self.reward_std + alpha * (reward - self.reward_mean) ** 2

        # Normalize the reward (z-score normalization)
        normalized_reward = (reward - self.reward_mean) / (self.reward_std ** 0.5)

        # Use tanh to squash the normalized reward to [-1, 1]
        with torch.no_grad():
            normalized_reward_tensor = torch.tensor(normalized_reward, dtype=torch.float32).to(self.device)
            scaled_reward = torch.tanh(normalized_reward_tensor)

        return scaled_reward

    def train(self, num_episodes, logging):
        episode_durations = []
        episode_rewards = []
        q_values = []

        opponent_ai = DQNAgent(device)
        opponent_ai.policy_net.load_state_dict(self.policy_net.state_dict())
        opponent_ai.target_net.load_state_dict(self.target_net.state_dict())

        for i in range(num_episodes):

            # initialize environment, equivalent to state, _ = env.reset()
            game = Game()
            player_paddle = game.get_paddle()
            opponent_paddle = game.get_other_paddle()
            ball = game.get_ball()
            episode_reward = 0
            q_value = []

            # we pack the state as the four desired inputs
            state = torch.tensor([ball.frect.pos[0], ball.frect.pos[1], player_paddle.frect.pos[0], player_paddle.frect.pos[1]], dtype=torch.float32, device=self.device)

            # this loop takes 0.002 second per timesteps
            # approx each episode is 2000 timesteps
            for t in count():

                opponent_move = opponent_ai.select_action(state)
                action = self.select_action(state) # -1, 0, 1 for self, steps down ++

                opponent_paddle.set_move(opponent_move.item())
                player_paddle.set_move(action.item())

                run, reward = self.get_reward(game) # this updates the state
                reward = self.normalize_reward(reward)
                episode_reward += reward
                step_reward = torch.tensor([reward], device=self.device, dtype=torch.float32).unsqueeze(0)

                # now we can read the next state
                obs = torch.tensor([ball.frect.pos[0], ball.frect.pos[1], player_paddle.frect.pos[0], player_paddle.frect.pos[1]], dtype=torch.float32, device=self.device)

                if not run:
                    next_state = None
                else:
                    next_state = obs

                self.memory.push(state, action, next_state, step_reward)
                state = next_state

                q = self.optimize_model()  # Assume optimize_model returns the Q-value of the chosen action
                if q is not None:
                    q_value.append(q)  # Record Q-value for plotting

                # now we use soft update to update the two networks
                if self.steps_done % TARGET_UPDATE_FREQ == 0:
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in target_net_state_dict:
                        target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU) * target_net_state_dict[key]
                    self.target_net.load_state_dict(target_net_state_dict)

                if self.steps_done % OPP_UPDATE_FREQ == 0:
                    opponent_ai.policy_net.load_state_dict(self.policy_net.state_dict())
                    opponent_ai.target_net.load_state_dict(self.target_net.state_dict())

                if not run:
                    episode_durations.append(t + 1)
                    episode_rewards.append(episode_reward)
                    q_values.append(sum(q_value) / len(q_value) if q_value else 0)
                    logging.info(f"Episode {i} completed in {t + 1} steps with reward {episode_reward}")
                    break
            
            print(f"Episode {i} completed in {t + 1} steps with reward {episode_reward}")
        
        torch.save(self.policy_net.state_dict(), "models/pong_dqn.pth")
        print("Model saved as pong_dqn.pth")

        return episode_durations, episode_rewards, q_values


    def load_weights(self, path="models/pong_dqn.pth") -> None:
        self.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def inference(self, state: tuple) -> int:
        with torch.inference_mode():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action_values = self.policy_net(state_tensor.unsqueeze(0))
            action = action_values.max(1)[1] - 1  # Maps 0,1,2 to -1,0,1
            return action.item() 
        

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    '''
    return "up" or "down", depending on which way the paddle should go to
    '''

    output = agent.inference((ball_frect.pos[0], ball_frect.pos[1], 
                              paddle_frect.pos[0], paddle_frect.pos[1]))
    return None if output == 0 else "up" if output == 1 else "down"


def plot_training_results(episode_rewards, q_values):

    import matplotlib.pyplot as plt

    # Plot reward over episodes
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode')

    # Plot Q-values over time
    plt.subplot(1, 2, 2)
    plt.plot(q_values)
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.title('Q-value over Time')

    plt.tight_layout()
    plt.show()


if __name__ != "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device)
    agent.load_weights()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device)

    import logging
    logging.basicConfig(filename="training.log", level=logging.INFO)

    episodes = 250 # Adjust as needed

    durations, rewards, q_values = agent.train(episodes, logging)

    plot_training_results(rewards, q_values)

    print(f"Training completed over {episodes} episodes")
    

