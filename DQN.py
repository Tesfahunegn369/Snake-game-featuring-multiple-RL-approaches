import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
import random
import copy
# Usage
input_size = 11  # Example input size
hidden_size = 256  # Example hidden layer size
output_size = 3  # Example output size
lr =0.001  # Learning rate
gamma = 0.99  # Discount factor
target_update = 10  # How often to update the target network
batch_size = 32  # Batch size for experience replay
max_memory_size = 10000  # Maximum number of experiences we're saving

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class DQNTrainer:
    def __init__(self, model, lr, gamma, target_update, batch_size, max_memory_size):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=max_memory_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, states, actions, rewards, next_states, dones):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        current_q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q = self.target_model(next_states).max(1)[0].detach()
        expected_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.criterion(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.steps_done += 1



model = DQN(input_size, hidden_size, output_size)
trainer = DQNTrainer(model, lr, gamma, target_update, batch_size, max_memory_size)

# Example of how to train
# trainer.remember(state, action, reward, next_state, done)
# trainer.train_step()
