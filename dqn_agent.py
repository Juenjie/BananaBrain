import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # soft update of target net parameter
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # update the net parameter every for steps

# set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed):
        """Interacts with and learns from the environment

        Params
        ======
              state_size (int): dimension of each state
              action_size (int): dimension of each action
              seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Netwrok
        self.qnetwork_local = QNetwork(state_size,action_size,seed).to(device)
        self.qnetwork_target = QNetwork(state_size,action_size,seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for ipdating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # only learn when enough samples have been stored
            # for every UPDATE_EVERY time steps
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples
        
        Params
        ======
              experiences (Tuple[torch.Tensor]): tuple of (s,a,r,s',done)
              gamma (float): discount factor"""
        
        states, actions, rewards, next_states, dones = experiences
        
        # Q-target net gives the Q-value of next_states
        Qsa_max = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # y, in the deep-Q paper
        y = rewards + gamma*Qsa_max*(1-dones)
        
        # loss function
        criterion = torch.nn.SmoothL1Loss()
        
        num_epoches = 10
        
        for epoch in range(num_epoches):
            # Q value from local net needs to be trained to catch up the y value
            Qsa_local = torch.gather(self.qnetwork_local(states),1,actions)
            
            loss = criterion(Qsa_local, y)
            
            # each training must set zero parameter gradients, 
            # otherwise the gradients will be accumulated automatically.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # update target network
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target
        
        Params
        ======
              local_model (PyTorch model): whose weights will be copied to target
              target_model (PyTorch model): whose weights will be changed
              tau (float): interpolation parameter
              """
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau*local_param.data+(1.-tau)*target_param.data)
            
    def act(self, state, eps = 0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
              state (array_like): current state
              eps (float): epsilon, for epsilon-greedy action selection
        """
        # use unsqueeze to add a branket
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # set the net to evaluation mode
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)