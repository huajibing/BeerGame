import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the DQN model (Q-Network)
class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        max_order=20,  # Maximum order quantity
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9999,
        buffer_size=10000,
        batch_size=64,
        update_every=4
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.max_order = max_order
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_every = update_every
        self.step_count = 0
        
        # Create Q networks (policy and target)
        self.policy_net = DQNNetwork(state_size, action_size)
        self.target_net = DQNNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Set device (CPU or GPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        # Q-losses list
        self.q_losses = []
    
    def act(self, state, eval_mode=False):
        """Select an action using epsilon-greedy policy"""
        if not eval_mode and random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(1, self.max_order + 1)
        else:
            # Exploitation: use policy network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
            self.policy_net.train()
            
            # Add 1 to convert from 0-index to 1-based order quantities
            return np.argmax(action_values.cpu().data.numpy()) + 1
    
    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and learn if it's time"""
        # Add experience to memory
        self.memory.add(state, action - 1, reward, next_state, done)  # Subtract 1 to convert to 0-index
        
        # Increment step counter
        self.step_count += 1
        
        # Learn every update_every steps when enough samples are available
        if len(self.memory) > self.batch_size and self.step_count % self.update_every == 0:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def learn(self, experiences):
        """Update policy network using batch of experiences"""
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        # Get expected Q values from policy network
        Q_expected = self.policy_net(states).gather(1, actions)
        
        # Get target Q values from target network
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute target Q values
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record Q-loss
        self.q_losses.append(loss.item())

        # Update target network
        if self.step_count % (self.update_every * 10) == 0:
            self.update_target()
    
    def update_target(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_average_q_loss(self):
        """Calculate and return the average Q-loss, then clear the list."""
        if not self.q_losses:
            return 0.0

        avg_loss = np.mean(self.q_losses)
        self.q_losses = []
        return avg_loss