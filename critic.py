import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(3 * embedding_dim, embedding_dim)
        self.concat_fc = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        s = torch.relu(self.fc1(x[1]))
        s = torch.cat([x[0], s], dim=1)
        s = torch.relu(self.concat_fc(s))
        s = torch.relu(self.fc2(s))
        s = torch.relu(self.fc3(s))
        return self.out(s)

class Critic:
    
    def __init__(self, hidden_dim, learning_rate, embedding_dim, tau):
        
        self.embedding_dim = embedding_dim

        # Critic network and target network
        self.network = CriticNetwork(embedding_dim, hidden_dim)
        self.target_network = CriticNetwork(embedding_dim, hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        # Loss function
        self.loss_fn = nn.MSELoss(reduction='none')

        # Soft target network update hyperparameter
        self.tau = tau

    def build_networks(self):
        # Build networks (dummy forward pass to initialize)
        self.network([torch.zeros(1, self.embedding_dim), torch.zeros(1, 3 * self.embedding_dim)])
        self.target_network([torch.zeros(1, self.embedding_dim), torch.zeros(1, 3 * self.embedding_dim)])
    
    def update_target_network(self):
        # Soft target network update
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def dq_da(self, inputs):
        actions = inputs[0]
        states = inputs[1]
        actions = torch.tensor(actions, requires_grad=True)
        outputs = self.network([actions, states])
        outputs.backward(torch.ones_like(outputs))
        return actions.grad

    def train(self, inputs, td_targets, weight_batch):
        weight_batch = torch.tensor(weight_batch, dtype=torch.float32)
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        
        loss = self.loss_fn(outputs, torch.from_numpy(td_targets))
        weighted_loss = (loss * weight_batch).mean()
        weighted_loss.backward()
        self.optimizer.step()
        return weighted_loss.item()

    def train_on_batch(self, inputs, td_targets, weight_batch):
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = self.loss_fn(outputs, td_targets)
        weighted_loss = (loss * weight_batch).mean()
        weighted_loss.backward()
        self.optimizer.step()
        return weighted_loss.item()
            
    def save_weights(self, path):
        torch.save(self.network.state_dict(), path)
        
    def load_weights(self, path):
        self.network.load_state_dict(torch.load(path))
