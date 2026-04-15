import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Example usage:
input_size = 11  # Example input size
hidden_size = 256  # Example hidden size
output_size = 3  # Example output size
lr = 0.001  # Learning rate
gamma = 0.99  # Discount factor
class Linear_QNet(nn.Module):
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


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        with torch.no_grad():
            next_pred = self.model(next_state)
            next_target = self.target_model(next_state)
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    best_next_action = torch.argmax(next_pred[idx]).item()
                    Q_new += self.gamma * next_target[idx][best_next_action]
                target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()



# Create the main Q-network and target Q-network
model = Linear_QNet(input_size, hidden_size, output_size)
target_model = Linear_QNet(input_size, hidden_size, output_size)

# Initialize the QTrainer
trainer = QTrainer(model, target_model, lr, gamma)

# Training loop:
# Inside the training loop, call trainer.train_step() with appropriate arguments to perform a training step.
# Ensure to update the target network periodically.
