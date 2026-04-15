import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        state_values = self.critic(x)
        return action_probs, state_values

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class PPOTrainer:
    def __init__(self, model, lr, gamma, eps_clip, K_epochs):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def train_step(self, states, actions, rewards, next_states, dones, old_log_probs):
        # Convert lists to tensor
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            log_probs, state_values = self.model(states)
            dist = torch.distributions.Categorical(log_probs)
            new_log_probs = dist.log_prob(actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(new_log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + F.mse_loss(state_values, rewards)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
