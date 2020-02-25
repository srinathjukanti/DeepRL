import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        hidden_state = F.relu(self.conv1(state))
        hidden_state = F.relu(self.conv2(hidden_state))
        hidden_state = F.relu(self.conv3(hidden_state))

        fc_input = hidden_state.view(hidden_state.size()[0], -1)
        hidden_state = F.relu(self.fc1(fc_input))
        actions = self.fc2(hidden_state)

        return actions

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        hidden_state = self.conv1(state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv3(hidden_state)

        return int(np.prod(hidden_state.size()))

    def save_checkpoint(self):
        print('##.....Saving checkpoint.....##')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('##.....Loading checkpoint.....##')
        self.load_state_dict(T.load(self.checkpoint_file))
        
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super(DuelingDeepQNetwork, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        hidden_state = F.relu(self.conv1(state))
        hidden_state = F.relu(self.conv2(hidden_state))
        hidden_state = F.relu(self.conv3(hidden_state))

        fc_input = hidden_state.view(hidden_state.size()[0], -1)
        hidden_state = F.relu(self.fc1(fc_input))

        A = self.A(hidden_state)
        V = self.V(hidden_state)

        return V, A

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        hidden_state = self.conv1(state)
        hidden_state = self.conv2(hidden_state)
        hidden_state = self.conv3(hidden_state)

        return int(np.prod(hidden_state.size()))

    def save_checkpoint(self):
        print('##.....Saving checkpoint.....##')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('##.....Loading checkpoint.....##')
        self.load_state_dict(T.load(self.checkpoint_file))