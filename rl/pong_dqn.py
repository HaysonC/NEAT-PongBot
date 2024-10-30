import random, math, json
from collections import deque, namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import gymnasium as gym
import ale_py

import logging
logging.basicConfig(filename="training.log", level=logging.INFO)

# ======================Replay Buffer=======================

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Simple Q-Network using CNN

class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(22528, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent():
    def __init__(self, env, device):
        num_actions = env.action_space.n

        self.policy_net = QNetwork(num_actions).to(device)
        self.target_net = QNetwork(num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.env = env
        self.device = device
        self.num_actions = num_actions
        self.policy_net = self.policy_net
        self.target_net = self.target_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.num_steps_to_update = 1000
    
    def select_action(self, state):
        eps_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1. * self.steps_done / 200)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.inference_mode():
                action_values = self.policy_net(state)
                action = action_values.max(1)[1] # batching happens
                return torch.tensor([action[0]], device=self.device, dtype=torch.int32) # only most recent action
        else:
            return torch.tensor([self.env.action_space.sample()], device=self.device, dtype=torch.int32)
    
    def optimize_model(self):
        if len(self.memory) < 32 or self.steps_done % self.num_steps_to_update != 0:
            return
        
        transitions = self.memory.sample(32)
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())

        next_state_values = torch.zeros(32, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * 0.99) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

    def train(self, num_episodes):
        episode_durations = []

        for i in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).permute(2, 0, 1)
            for t in count():
                action = self.select_action(state.unsqueeze(0))
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).permute(2, 0, 1)

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in target_net_state_dict:
                    target_net_state_dict[key] = 0.005 * policy_net_state_dict[key] + (1 - 0.005) * target_net_state_dict[key]
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    break

                logging.info(f"Episode {i} completed in {t + 1} steps with reward {reward.item()}")
        
        torch.save(self.policy_net.state_dict(), "../models/pong_dqn.pth")
        print("Model saved as qnetwork_trained.pth")

        return episode_durations