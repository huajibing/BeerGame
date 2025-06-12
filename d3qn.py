import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define Dueling DQN network architecture
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DuelingDQNNetwork, self).__init__()
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream - estimates state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream - estimates advantage for each action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage to get Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

# SumTree for Prioritized Experience Replay
# Adapted from OpenAI Baselines
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (experiences)
        self.tree = np.zeros(2 * capacity - 1)  # Full binary tree, stores priorities
        self.data = np.zeros(capacity, dtype=object)  # Stores experience objects
        self.data_pointer = 0 # Current position to write new data
        self.n_entries = 0 # Current number of entries in the tree

    # Update parent nodes when a leaf node's priority changes
    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # Update priority of a leaf node
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    # Add a new experience (priority and data)
    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1 # Leaf node index
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity: # Reset pointer if capacity is reached
            self.data_pointer = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # Find sample index for a given priority sum 's'
    def _retrieve(self, tree_index, s):
        left_child = 2 * tree_index + 1
        right_child = left_child + 1

        if left_child >= len(self.tree): # Reached leaf node
            return tree_index

        if s <= self.tree[left_child]:
            return self._retrieve(left_child, s)
        else:
            return self._retrieve(right_child, s - self.tree[left_child])

    # Get total priority (root node value)
    def total_priority(self):
        return self.tree[0]

    # Get a sample (tree_index, priority, data)
    def get_leaf(self, s):
        tree_index = self._retrieve(0, s)
        data_index = tree_index - self.capacity + 1
        return tree_index, self.tree[tree_index], self.data[data_index]

    def __len__(self):
        return self.n_entries

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, epsilon_for_priority=1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Controls how much prioritization is used (0: uniform, 1: full)
        self.epsilon_for_priority = epsilon_for_priority # Small constant to ensure non-zero priority
        self.max_priority = 1.0 # Initial max priority for new experiences

    def add(self, *experience): # state, action, reward, next_state, done
        # New experiences are added with max priority to ensure they are sampled at least once
        priority = self.max_priority 
        self.tree.add(priority, experience)

    def sample(self, batch_size, beta=0.4):
        experiences = []
        indices = np.empty((batch_size,), dtype=np.int32)
        is_weights = np.empty((batch_size,), dtype=np.float32)
        
        segment_priority = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            # Sample a value from each segment
            a = segment_priority * i
            b = segment_priority * (i + 1)
            s = random.uniform(a, b)
            
            # Retrieve the experience
            tree_idx, priority, data = self.tree.get_leaf(s)
            
            experiences.append(data)
            indices[i] = tree_idx
            
            # Calculate importance sampling weight
            sampling_probability = priority / self.tree.total_priority()
            is_weights[i] = (self.tree.n_entries * sampling_probability) ** (-beta)
            
        # Normalize IS weights by the maximum weight for stability
        if is_weights.max() > 0:
             is_weights /= is_weights.max()
        else: # Handle case where all weights might be zero (e.g., if total_priority is zero)
             is_weights = np.ones_like(is_weights)


        # Unzip experiences
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones, indices, is_weights

    def update_priorities(self, tree_indices, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon_for_priority) ** self.alpha
        # Ensure priorities are positive
        priorities = np.maximum(priorities, self.epsilon_for_priority) 

        for idx, priority in zip(tree_indices, priorities):
            self.tree.update(idx, priority)
        
        # Update max_priority if any new priority is higher
        current_max = np.max(priorities)
        if current_max > self.max_priority and current_max < 1e6 : # Avoid extremely large priorities
            self.max_priority = current_max


    def __len__(self):
        return len(self.tree)

# Standard Experience Replay Buffer (for when PER is not used)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones, None, np.ones(batch_size) # Return dummy indices and weights
    
    def __len__(self):
        return len(self.buffer)
    
    def update_priorities(self, tree_indices, td_errors): # Dummy for compatibility
        pass

# D3QN Agent (Double Dueling DQN) with PER
class D3QNAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        max_order=20,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9999,
        buffer_size=10000,
        batch_size=64,
        update_every=4,
        tau=0.005,
        prioritized_replay=False, # Flag to enable/disable PER
        alpha_per=0.6, # PER exponent for priorities
        beta_per_start=0.4, # Initial PER importance sampling exponent
        beta_per_end=1.0, # Final PER importance sampling exponent
        beta_per_frames=100000 # Number of frames to anneal beta to beta_per_end
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
        self.tau = tau
        self.step_count = 0
        
        self.prioritized_replay = prioritized_replay
        
        # Create Q networks (policy and target)
        self.policy_net = DuelingDQNNetwork(state_size, action_size)
        self.target_net = DuelingDQNNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network in evaluation mode
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        if self.prioritized_replay:
            print("Using Prioritized Replay Buffer")
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha=alpha_per)
            self.beta = beta_per_start
            self.beta_end = beta_per_end
            # Calculate beta increment per sampling step, ensuring it doesn't exceed beta_end
            # Total number of learning steps (samples) can be estimated.
            # If beta_per_frames is total environment steps, then learning steps are beta_per_frames / update_every
            num_learning_steps_for_beta_anneal = beta_per_frames / update_every
            if num_learning_steps_for_beta_anneal > 0:
                 self.beta_increment = (beta_per_end - beta_per_start) / num_learning_steps_for_beta_anneal
            else:
                 self.beta_increment = 0 # Beta will not anneal if beta_per_frames is too small or zero
        else:
            print("Using Standard Replay Buffer")
            self.memory = ReplayBuffer(buffer_size)
            self.beta = 1.0 # No IS correction needed for standard replay
            self.beta_increment = 0

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
            self.policy_net.eval() # Set to evaluation mode for inference
            with torch.no_grad():
                action_values = self.policy_net(state_tensor)
            self.policy_net.train() # Set back to training mode
            
            # Add 1 to convert from 0-index to 1-based order quantities
            return np.argmax(action_values.cpu().data.numpy()) + 1
    
    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and learn if it's time"""
        # Add experience to memory
        # For PER, initial TD error is not known here, so add with max priority
        # The priority will be updated after the first learning step involving this transition.
        self.memory.add(state, action - 1, reward, next_state, done) # Subtract 1 to convert to 0-index for action
        
        # Increment step counter
        self.step_count += 1
        
        # Learn every update_every steps when enough samples are available
        if len(self.memory) > self.batch_size and self.step_count % self.update_every == 0:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Anneal beta for PER
            if self.prioritized_replay:
                self.beta = min(self.beta_end, self.beta + self.beta_increment)
    
    def learn(self, experiences):
        """Update policy network using batch of experiences (with PER if enabled)"""
        states, actions, rewards, next_states, dones, tree_indices, is_weights = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device) # Actions are 0-indexed
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        is_weights_tensor = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # Get Q-values for current states from policy_net: Q(s_t, a_t)
        Q_expected = self.policy_net(states).gather(1, actions)
        
        # Double DQN:
        # 1. Get best actions for next_states from policy_net: a'_max = argmax_a' Q_policy(s_{t+1}, a')
        with torch.no_grad(): # No gradient calculation for this part
            best_actions_next = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        # 2. Get Q-values for these best_actions_next from target_net: Q_target(s_{t+1}, a'_max)
            Q_targets_next = self.target_net(next_states).gather(1, best_actions_next)
        
        # Compute target Q-values: R_t + gamma * Q_target(s_{t+1}, a'_max) * (1 - done)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Calculate TD errors for PER priority update
        td_errors = (Q_expected - Q_targets).detach().cpu().numpy().flatten() # Get raw TD errors

        # Update priorities in PER buffer
        if self.prioritized_replay and tree_indices is not None:
            self.memory.update_priorities(tree_indices, td_errors)
        
        # Compute loss: (IS_weight * (Q_expected - Q_targets)^2).mean()
        # Using Huber loss (SmoothL1Loss) can also be beneficial for stability
        # For PER, loss is weighted by IS weights
        loss = (is_weights_tensor * nn.functional.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
        # loss = (is_weights_tensor * nn.functional.smooth_l1_loss(Q_expected, Q_targets, reduction='none')).mean()


        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Record Q-loss
        self.q_losses.append(loss.item())

        # Soft update target network
        self.soft_update()
    
    def soft_update(self):
        """Soft update target network parameters: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def get_average_q_loss(self):
        """Calculate and return the average Q-loss, then clear the list."""
        if not self.q_losses:
            return 0.0

        avg_loss = np.mean(self.q_losses)
        self.q_losses = []
        return avg_loss