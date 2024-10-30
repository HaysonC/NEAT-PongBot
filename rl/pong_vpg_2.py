# Environment: https://ale.farama.org/environments/pong/

# NOTE: Frame Stacking is currently done with 3D CNN, 
# but realistically an RNN wrapper would do better.

# Estimated: 65,426,084 parameters per network

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import ale_py

# Frame stacking class
# NOTE: this is a wrapper for env calls
class FrameStack:
    def __init__(self, env, n_frames):
        self.env = env
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self):
        state, _ = self.env.reset()
        # Initialize the frame stack with the first frame
        for _ in range(self.n_frames):
            self.frames.append(state)
        return np.array(self.frames)

    def step(self, action):
        state, reward, done, _, _ = self.env.step(action)
        self.frames.append(state)
        return np.array(self.frames), reward, done

# CNN-based Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__() # input size: (4, 3, 210, 160)
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(2, 4, 4), stride=(1, 2, 2))  # Adjusted kernel size and stride
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(2, 4, 4), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(2, 3, 3), stride=(1, 1, 1))
        self.fc1 = nn.Linear(112896, 512)  # Update the input size based on CNN output
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        return Categorical(logits=self.fc2(x))  # Sample from the categorical distribution

# CNN-based Value Network
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__() # Change input channels to 3 (RGB)
        '''
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4) # Output: (batch, 32, 51, 39)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # (batch, 64, 24, 18)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # (batch, 64, 22, 16)
        '''
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(2, 4, 4), stride=(1, 2, 2))  # Adjusted kernel size and stride
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(2, 4, 4), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(2, 3, 3), stride=(1, 1, 1))
        self.fc1 = nn.Linear(112896, 512)  # Update the input size based on CNN output: 64 * 22 * 16
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class VPGAgent:
    def __init__(self, action_dim, n_frames=4, lr=1e-2, gamma=0.9, lam=0.95):
        self.policy_net = PolicyNetwork(action_dim)
        self.value_net = ValueNetwork()
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam # lambda parameter for GAE
        self.n_frames = n_frames

    def select_action(self, obs):
        obs = obs.transpose(3, 0, 1, 2)
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.unsqueeze(0)
        dist = self.policy_net(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy
    
    # Generalized Advantage Estimation
    def compute_gae(self, rewards, values, dones): # values are estimated by value net
        advantages = []
        gae = 0
        next_value = 0 # at end next value must be 0

        for i in reversed(range(len(rewards))):
            # when you interate from end it's basically recursion
            if dones[i]:
                next_value = 0
            delta = rewards[i] + self.gamma * next_value - values[i] # TD: r(t) + gamma * V(t+1) - V(t)
            # we recursivley update gae, using lambda to control variance-bias tradeoff
            gae = delta + self.gamma * self.lam * gae # A(i) = delta(i) + gamma * lambda * delta(i+1) + (gamma * lambda)^2 * delta(i+2) +...
            advantages.insert(0, gae) # note that you are building this from the end to maintain correct order
            next_value = values[i]
        
        returns = [adv + val for adv, val in zip(advantages, values)] # by definition
        return advantages, returns # adv updates policy net and returns updates value function
    
    def update(self, trajectories):
        obs = torch.tensor(np.array([t[0] for t in trajectories]).transpose(0, 4, 1, 2, 3), dtype=torch.float32)
        acts = torch.tensor(np.array([t[1] for t in trajectories]), dtype=torch.int32)
        rewards = [t[2] for t in trajectories]
        log_probs = torch.stack([t[3] for t in trajectories])
        dones = [t[4] for t in trajectories]
        entropies = torch.stack([t[5] for t in trajectories])

        values = self.value_net(obs).squeeze().detach().numpy()
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # UPDATE POLICY
        self.policy_optimizer.zero_grad()
        # See notes and simple_pg for loss fn, added entropy regularization
        policy_loss = -(log_probs * advantages).mean() - 0.01 * entropies.mean() 
        policy_loss.backward()
        self.policy_optimizer.step()

        # UPDATE VALUE FUNCTION
        self.value_optimizer.zero_grad()
        values = self.value_net(obs).squeeze() # Simple MSE loss
        value_loss = nn.functional.mse_loss(values, returns)
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, env: gym.Env, num_episodes: int):

        frame_stack = FrameStack(env, self.n_frames)

        for e in range(num_episodes):
            obs = frame_stack.reset() # reset and get initial frame stack
            trajectories = []
            episode_reward = 0
            done = False

            while not done:
                action, log_prob, entropy = self.select_action(obs)
                next_state, reward, done, = frame_stack.step(action)

                trajectories.append((obs, action, reward, log_prob, done, entropy))
                episode_reward += reward
                obs = next_state
            
            self.update(trajectories)
            print(f"Episode {e+1}: Total Reward: {episode_reward}")

        torch.save(self.policy_net.state_dict(), "../models/pong_vpg.pth")
    
    def load_model(self, state_dict_path="../models/vpg.pth"):
        self.policy_net.load_state_dict(torch.load(state_dict_path, weights_only=True))
        self.policy_net.eval()

    def inference(self, env: gym.Env, episode=10, render=True) -> list:
    
        episode_rewards = []  # List to store total rewards for each episode
        
        for _ in range(episode):

            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                env.render()
                # Convert state to tensor and pass it to the model to get the action
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.inference_mode():
                    action = self.policy_net(state_tensor).sample().item()  # Select best action
                
                state, reward, done, _, _ = env.step(action)  # Take action in the environment
                total_reward += reward
                
                if render:
                    env.render()  # Render the environment if specified

            episode_rewards.append(total_reward)
            print(f'Total Reward: {total_reward}')
        
        env.close()  # Close the environment
        
        # Plot the rewards per episode
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid()
        plt.show()
        
        return episode_rewards

