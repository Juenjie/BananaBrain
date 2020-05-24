import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model"""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimensions of each state
            action_size (int): Dimensions of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # construct the model
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps states -> Q(s,a)."""
        
        layer = F.leaky_relu(self.fc1(state))
        layer = F.leaky_relu(self.fc2(layer))
        layer = F.leaky_relu(self.fc3(layer))
        layer = self.fc4(layer)
        
        return layer
        
        
        
        
        