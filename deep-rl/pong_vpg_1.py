# Environment: https://ale.farama.org/environments/pong/

# Estimated: 11,612,836 parameter per network

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import ale_py

# CNN-based Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # Adjusted kernel size and stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(22528, 512)  # Update the input size based on CNN output
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
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # Adjusted kernel size and stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(22528, 512)  # Update the input size based on CNN output
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class VPGAgent:
    def __init__(self, action_dim, lr=1e-1, gamma=0.8, lam=0.95):
        self.policy_net = PolicyNetwork(action_dim)
        self.value_net = ValueNetwork()
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam  # lambda parameter for GAE

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        dist = self.policy_net(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy

    # Generalized Advantage Estimation
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = 0  # at end next value must be 0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            next_value = values[i]

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self, trajectories):
        obs = torch.tensor(np.array([t[0] for t in trajectories]), dtype=torch.float32).permute(0, 3, 1, 2)
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
        policy_loss = -(log_probs * advantages).mean() - 0.01 * entropies.mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        # UPDATE VALUE FUNCTION
        self.value_optimizer.zero_grad()
        values = self.value_net(obs).squeeze()
        value_loss = nn.functional.mse_loss(values, returns)
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, env: gym.Env, num_episodes: int):
        for e in range(num_episodes):
            obs, _ = env.reset()
            trajectories = []
            episode_reward = 0
            done = False

            while not done:
                action, log_prob, entropy = self.select_action(obs)
                next_state, reward, done, _, _ = env.step(action)

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
        episode_rewards = []

        for _ in range(episode):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                if render:
                    env.render()
                state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)
                with torch.inference_mode():
                    action = self.policy_net(state_tensor).sample().item()

                state, reward, done, _, _ = env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)
            print(f'Total Reward: {total_reward}')

        env.close()

        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid()
        plt.show()

        return episode_rewards