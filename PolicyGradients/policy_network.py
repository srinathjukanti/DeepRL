import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class ReinforcePolicyNet(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, checkpoint_dir):
        super(ReinforcePolicyNet, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = nn.Linear(input_dims[0], 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        hidden_state = F.relu(self.fc1(state))
        hidden_state = F.relu(self.fc2(hidden_state))
        actions = F.relu(self.fc3(hidden_state))

        return F.softmax(actions)

    def save_checkpoint(self):
        print('##.....Saving checkpoint.....##')
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('##.....Loading checkpoint.....##')
        self.load_state_dict(T.load(self.checkpoint_file))