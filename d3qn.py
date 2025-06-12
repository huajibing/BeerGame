import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import random
from collections import deque

# Define NoisyLinear layer for Noisy Networks
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters for weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        # Learnable parameters for biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Non-learnable buffers for noise (sampled during training)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialization similar to nn.Linear
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features)) # Using out_features for bias sigma like some implementations

    def reset_noise(self):
        # Sample noise from a standard normal distribution
        # Factorized Gaussian noise:
        # For weights: epsilon_w_ij = f(epsilon_i) * f(epsilon_j)
        # For biases: epsilon_b_j = f(epsilon_j)
        # where f(x) = sgn(x) * sqrt(|x|)
        # Simpler approach: Independent Gaussian noise (used here)
        self.weight_epsilon.data.normal_()
        self.bias_epsilon.data.normal_()

    def forward(self, input):
        if self.training:
            # Noisy weights and biases for training
            weight = self.weight_mu + (self.weight_sigma * self.weight_epsilon)
            bias = self.bias_mu + (self.bias_sigma * self.bias_epsilon)
        else:
            # Deterministic weights and biases for evaluation
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

# Define Dueling DQN network architecture
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, noisy_std_init=0.1, lstm_hidden_size=64,
                 num_atoms=51, v_min=-10, v_max=10): # C51 params
        super(DuelingDQNNetwork, self).__init__()
        self.noisy_std_init = noisy_std_init
        self.lstm_hidden_size = lstm_hidden_size
        self.action_size = output_size # Renaming for clarity in C51 context
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))

        # Shared feature layer (processes individual states in a sequence)
        self.feature_layer = nn.Sequential(
            NoisyLinear(input_size, hidden_size, std_init=self.noisy_std_init),
            nn.ReLU()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, self.lstm_hidden_size, batch_first=True)

        # Value stream - outputs distribution for state value V(s)
        self.value_stream = nn.Sequential(
            NoisyLinear(self.lstm_hidden_size, self.lstm_hidden_size, std_init=self.noisy_std_init),
            nn.ReLU(),
            NoisyLinear(self.lstm_hidden_size, self.num_atoms, std_init=self.noisy_std_init) # Outputs num_atoms
        )
        
        # Advantage stream - outputs distribution for advantage A(s,a) for each action
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.lstm_hidden_size, self.lstm_hidden_size, std_init=self.noisy_std_init),
            nn.ReLU(),
            NoisyLinear(self.lstm_hidden_size, self.action_size * self.num_atoms, std_init=self.noisy_std_init) # Outputs action_size * num_atoms
        )
    
    def forward(self, x, hx=None):
        # x shape: (batch_size, sequence_length, state_size)
        batch_size, sequence_length, state_size = x.shape

        # Process each state in the sequence through the feature_layer
        # To do this, we reshape x to (batch_size * sequence_length, state_size)
        # then pass it through feature_layer, then reshape back.
        x_reshaped = x.reshape(batch_size * sequence_length, state_size)
        features_reshaped = self.feature_layer(x_reshaped) # (batch_size * sequence_length, hidden_size)

        # Reshape features back to (batch_size, sequence_length, hidden_size) for LSTM
        features_seq = features_reshaped.reshape(batch_size, sequence_length, -1)

        # Pass features through LSTM
        # hx is (h_init, c_init)
        lstm_out, hx_next = self.lstm(features_seq, hx)
        # lstm_out shape: (batch_size, sequence_length, lstm_hidden_size)
        # hx_next is (h_n, c_n)

        # Value stream output (logits for value distribution)
        value_logits = self.value_stream(lstm_out) # Shape: (batch_size, sequence_length, num_atoms)

        # Advantage stream output (logits for advantage distribution for each action)
        advantage_logits_flat = self.advantage_stream(lstm_out) # Shape: (batch_size, sequence_length, action_size * num_atoms)

        # Reshape advantage logits
        advantage_logits = advantage_logits_flat.view(batch_size, sequence_length, self.action_size, self.num_atoms)
        
        # Dueling combination for distributional RL
        # value_logits expanded: (batch_size, sequence_length, 1, num_atoms)
        # advantage_logits: (batch_size, sequence_length, action_size, num_atoms)
        value_logits_expanded = value_logits.unsqueeze(2)

        # Mean of advantage logits over actions
        adv_mean_logits = advantage_logits.mean(dim=2, keepdim=True) # Shape: (batch_size, sequence_length, 1, num_atoms)

        # Combine: Z(s,a) = Z_V(s) + (Z_A(s,a) - mean_a(Z_A(s,a)))
        dist_logits = value_logits_expanded + (advantage_logits - adv_mean_logits)
        # dist_logits shape: (batch_size, sequence_length, action_size, num_atoms)

        # Network returns raw logits. Softmax is applied by the agent where needed.
        return dist_logits, hx_next

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers."""
        for name, module in self.named_children():
            if isinstance(module, nn.Sequential):
                for layer_name, layer in module.named_children():
                    if isinstance(layer, NoisyLinear):
                        layer.reset_noise()
            elif isinstance(module, NoisyLinear):
                module.reset_noise()

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
        self.alpha = alpha
        self.epsilon_for_priority = epsilon_for_priority
        self.max_priority = 1.0

    # For LSTM, experience is (state_sequence, action, reward, next_state_sequence, done)
    def add(self, *experience):
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


        # Unzip experiences: state_seqs, actions, rewards, next_state_seqs, dones
        # Original was: states, actions, rewards, next_states, dones = zip(*experiences)
        # Now, each 'data' in experiences is a tuple (state_sequence, action, reward, next_state_sequence, done)

        # Unzipping structure:
        # experience_tuples = [exp1, exp2, ..., exp_batch_size] where exp_i = (s_seq, a, r, ns_seq, d)
        # This means states_seq will be a tuple of state_sequences, actions a tuple of actions, etc.
        states_seq, actions, rewards, next_states_seq, dones = zip(*experiences)

        return states_seq, actions, rewards, next_states_seq, dones, indices, is_weights

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
    
    # For LSTM, experience is (state_sequence, action, reward, next_state_sequence, done)
    def add(self, *experience): # state_sequence, action, reward, next_state_sequence, done
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Unzip experiences: state_seqs, actions, rewards, next_state_seqs, dones
        states_seq, actions, rewards, next_states_seq, dones = zip(*batch)
        return states_seq, actions, rewards, next_states_seq, dones, None, np.ones(batch_size)
    
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
        buffer_size=10000,
        batch_size=64,
        update_every=4,
        tau=0.005,
        prioritized_replay=False,
        alpha_per=0.6,
        beta_per_start=0.4,
        beta_per_end=1.0,
        beta_per_frames=100000,
        noisy_std_init=0.1, # Default for NoisyLinear std_init
        sequence_length=8,  # Default sequence length for LSTM
        lstm_hidden_size=128,# Default LSTM hidden size
        num_atoms=51,       # Default number of atoms for C51
        v_min=-10,          # Default min value for C51 support
        v_max=10            # Default max value for C51 support
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.max_order = max_order
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        self.step_count = 0
        self.noisy_std_init = noisy_std_init
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        
        # C51 parameters
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.prioritized_replay = prioritized_replay
        
        self.state_history = deque(maxlen=self.sequence_length)
        self.agent_lstm_h_c = None

        # Create Q networks (policy and target)
        # Pass C51 parameters to DuelingDQNNetwork
        self.policy_net = DuelingDQNNetwork(state_size, action_size,
                                            noisy_std_init=self.noisy_std_init,
                                            lstm_hidden_size=self.lstm_hidden_size,
                                            num_atoms=self.num_atoms, v_min=self.v_min, v_max=self.v_max)
        self.target_net = DuelingDQNNetwork(state_size, action_size,
                                            noisy_std_init=self.noisy_std_init,
                                            lstm_hidden_size=self.lstm_hidden_size,
                                            num_atoms=self.num_atoms, v_min=self.v_min, v_max=self.v_max)
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

        # C51 support, move to device once
        self.support = self.policy_net.support.to(self.device)


    def reset_episode_states(self):
        """Resets state history and LSTM hidden state for the agent at the start of an episode."""
        self.state_history.clear()
        self.agent_lstm_h_c = None
    
    def act(self, current_observation, eval_mode=False):
        """Select an action using the policy network with LSTM and noise for exploration."""

        # Create a temporary history for generating the current sequence
        temp_history = list(self.state_history)
        temp_history.append(current_observation)

        # Pad if history is shorter than sequence_length
        if len(temp_history) < self.sequence_length:
            padding = [np.zeros_like(current_observation) for _ in range(self.sequence_length - len(temp_history))]
            # padding = [temp_history[0] for _ in range(self.sequence_length - len(temp_history))] # Pad with first state
            current_sequence_list = padding + temp_history
        else:
            current_sequence_list = temp_history[-self.sequence_length:]

        input_seq_tensor = torch.FloatTensor(np.array(current_sequence_list)).unsqueeze(0).to(self.device) # (1, seq_len, state_size)

        h_init, c_init = None, None
        if self.agent_lstm_h_c:
            h_init, c_init = self.agent_lstm_h_c
            # Detach hidden state if not in eval_mode to prevent backprop through entire episode history
            # For eval_mode, we might also want to detach, or reset per eval episode via reset_episode_states
            if not eval_mode : # or eval_mode if hidden state is carried over during evaluation episodes
                 h_init = h_init.detach()
                 c_init = c_init.detach()
            # If eval_mode and we want fresh start for LSTM state each time, ensure agent_lstm_h_c is None
            # This is handled by reset_episode_states called externally before an eval episode.

        current_hx = (h_init, c_init) if h_init is not None and c_init is not None else None

        if eval_mode:
            self.policy_net.eval()
        else:
            self.policy_net.train()
            self.policy_net.reset_noise()

        with torch.no_grad():
            # Network returns distributional logits: (batch, seq_len, action_size, num_atoms)
            dist_logits_seq, hx_next = self.policy_net(input_seq_tensor, hx=current_hx)
            
        if not eval_mode:
            self.agent_lstm_h_c = hx_next

        # Get logits for the last step in the sequence: (1, action_size, num_atoms)
        dist_logits_last_step = dist_logits_seq[:, -1, :, :]
        # Convert logits to probabilities
        dist_probs_last_step = F.softmax(dist_logits_last_step, dim=2) # Softmax over atoms

        # Calculate expected Q-values for each action from the distribution
        # self.support is (num_atoms), needs to be (1, 1, num_atoms) or similar for broadcasting
        q_values = (dist_probs_last_step * self.support).sum(dim=2) # Shape: (1, action_size)

        return np.argmax(q_values.cpu().data.numpy()) + 1
    
    def step(self, state_that_led_to_action, action, reward, next_state_after_action, done):
        """Store experience sequence in replay memory and learn if it's time."""
        self.state_history.append(state_that_led_to_action)

        # Store sequence if state_history is full
        if len(self.state_history) == self.sequence_length:
            current_seq_array = np.array(list(self.state_history), dtype=np.float32)

            # Construct next_state_sequence for buffer
            # This is state_history shifted by one, with next_state_after_action at the end
            next_history_for_buffer = list(self.state_history)[1:]
            next_history_for_buffer.append(next_state_after_action)
            next_seq_array = np.array(next_history_for_buffer, dtype=np.float32)

            # Add to memory: (state_sequence, action, reward, next_state_sequence, done)
            # action is 0-indexed for buffer
            self.memory.add(current_seq_array, action - 1, reward, next_seq_array, done)

        if done:
            self.reset_episode_states() # Reset for the next episode

        self.step_count += 1
        
        # Learn every update_every steps when enough samples are available
        if len(self.memory) > self.batch_size and self.step_count % self.update_every == 0:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            # Epsilon decay is removed as NoisyNets handle exploration
            
            # Anneal beta for PER
            if self.prioritized_replay:
                self.beta = min(self.beta_end, self.beta + self.beta_increment)
    
    def learn(self, experiences):
        """Update policy network using batch of experiences (sequences for LSTM)."""
        # experiences: state_seqs, actions, rewards, next_state_seqs, dones, tree_indices, is_weights
        states_seq, actions, rewards, next_states_seq, dones, tree_indices, is_weights = experiences
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states_seq)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states_seq)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions)).to(self.device) # Shape (batch_size,) for direct use
        rewards_tensor = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device) # (batch, 1)
        dones_tensor = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device) # (batch, 1)
        is_weights_tensor = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        self.policy_net.train()
        self.target_net.train()

        # Current log-probabilities for actions taken
        # policy_net returns logits: (batch, seq_len, action_size, num_atoms)
        all_dist_logits_policy_seq, _ = self.policy_net(states_tensor, hx=None)
        dist_logits_policy_last_step = all_dist_logits_policy_seq[:, -1, :, :] # (batch, action_size, num_atoms)
        log_dist_policy = F.log_softmax(dist_logits_policy_last_step, dim=2) # Log-softmax over atoms
        
        # Gather the log_probs for the specific actions taken
        # actions_tensor is (batch_size), needs to be (batch_size, 1, num_atoms) for gather
        actions_exp = actions_tensor.view(-1, 1, 1).expand(-1, 1, self.num_atoms)
        log_probs_for_actions = log_dist_policy.gather(1, actions_exp).squeeze(1) # (batch, num_atoms)

        with torch.no_grad():
            # Double DQN:
            # 1. Get best actions for next_states from policy_net (based on expected Q-values)
            all_next_dist_logits_policy_seq, _ = self.policy_net(next_states_tensor, hx=None)
            next_dist_logits_policy_last = all_next_dist_logits_policy_seq[:, -1, :, :] # (batch, action_size, num_atoms)
            next_dist_probs_policy_last = F.softmax(next_dist_logits_policy_last, dim=2)
            next_q_values_policy = (next_dist_probs_policy_last * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2) # (batch, action_size)
            best_actions_next_indices = next_q_values_policy.max(1)[1] # (batch,)

            # 2. Get next state-action distributional logits from target_net for these best actions
            all_next_dist_logits_target_seq, _ = self.target_net(next_states_tensor, hx=None)
            next_dist_logits_target_last = all_next_dist_logits_target_seq[:, -1, :, :] # (batch, action_size, num_atoms)

            # Gather the distributions for the best_actions_next_indices
            best_actions_exp_target = best_actions_next_indices.view(-1, 1, 1).expand(-1, 1, self.num_atoms)
            next_dist_target_logits = next_dist_logits_target_last.gather(1, best_actions_exp_target).squeeze(1) # (batch, num_atoms)
            next_dist_target_probs = F.softmax(next_dist_target_logits, dim=1) # (batch, num_atoms)

            # 3. Project Bellman update onto the support atoms
            Tz = rewards_tensor + self.gamma * (1 - dones_tensor) * self.support.unsqueeze(0) # (batch, num_atoms)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)
            b = (Tz - self.v_min) / self.delta_z # (batch, num_atoms)
            l = b.floor().long()
            u = b.ceil().long()

            # Distribute probability of Tz
            m = torch.zeros(self.batch_size, self.num_atoms, device=self.device)

            # Corrected projection using batch indexing (more robust than offset flatten)
            for i in range(self.batch_size):
                # For each atom in the distribution being projected (next_dist_target_probs[i])
                # And for each atom in the support for Tz (Tz[i])
                # This loop is over batch elements, inner operations are vectorized over atoms
                l_i = l[i] # Shape: (num_atoms,)
                u_i = u[i] # Shape: (num_atoms,)
                b_i = b[i] # Shape: (num_atoms,)
                dist_probs_i = next_dist_target_probs[i] # Shape: (num_atoms,)

                # Atom indices must be within [0, num_atoms - 1]
                l_i = l_i.clamp(0, self.num_atoms - 1)
                u_i = u_i.clamp(0, self.num_atoms - 1)

                m[i].index_add_(0, l_i, dist_probs_i * (u_i.float() - b_i))
                m[i].index_add_(0, u_i, dist_probs_i * (b_i - l_i.float()))

        # Loss: KL divergence (cross-entropy form as target m is fixed)
        # loss = - sum(m_target * log_pred_dist_for_action)
        loss = - (m * log_probs_for_actions).sum(dim=1) # (batch_size,)
        
        td_errors = loss.detach().abs().cpu().numpy() # For PER

        # Apply IS weights and mean reduction
        loss = (is_weights_tensor.squeeze(1) * loss).mean()

        self.optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # After learning, ensure target_net is back in evaluation mode for stability if it was changed.
        # The primary role of target_net.eval() is to disable dropout or batchnorm,
        # but for NoisyLinear, it means using mean weights.
        self.target_net.eval()
        
        # Soft update target network
        self.soft_update()

    def soft_update(self):
        """Soft update target network parameters: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        # Ensure target net is in eval mode before copying params,
        # though this primarily affects layers like Dropout/BatchNorm, not NoisyLinear's params themselves.
        self.target_net.eval()
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

if __name__ == '__main__':
    print("Starting D3QNAgent instantiation test...")

    # Basic parameters for testing
    state_s = 7  # Example state_size for EnvExtended
    action_s = 20 # Example action_size (max_order)
    max_o = 20   # max_order, often same as action_s if actions are 1 to max_order

    # Hyperparameters for D3QNAgent
    noisy_std = 0.1
    seq_len = 4
    lstm_hidden = 64 # Adjusted to a common value, can be 32 for quicker test too
    n_atoms = 51
    v_min_val = -10
    v_max_val = 10
    test_batch_size = 2 # Small batch size for testing learn step
    test_buffer_size = 10 # Small buffer

    print(f"State size: {state_s}, Action size: {action_s}, Max order: {max_o}")
    print(f"Sequence length: {seq_len}, LSTM hidden: {lstm_hidden}")
    print(f"Num atoms: {n_atoms}, V_min: {v_min_val}, V_max: {v_max_val}")

    try:
        # Instantiate D3QNAgent
        agent = D3QNAgent(
            state_size=state_s,
            action_size=action_s,
            max_order=max_o,
            prioritized_replay=True, # Test with PER
            noisy_std_init=noisy_std,
            sequence_length=seq_len,
            lstm_hidden_size=lstm_hidden,
            num_atoms=n_atoms,
            v_min=v_min_val,
            v_max=v_max_val,
            batch_size=test_batch_size,
            buffer_size=test_buffer_size, # Use small buffer for test
            alpha_per=0.6,            # Example PER alpha
            beta_per_start=0.4,       # Example PER beta start
            beta_per_end=1.0,         # Example PER beta end
            beta_per_frames=100       # Example PER beta frames (small for test)
        )
        print("D3QNAgent instantiated successfully.")

        # Test reset_episode_states
        agent.reset_episode_states()
        print("agent.reset_episode_states() called.")

        # Test act method
        dummy_state_obs = np.random.rand(state_s).astype(np.float32)
        action = agent.act(dummy_state_obs)
        print(f"agent.act(dummy_state_obs) returned action: {action}")

        # Test multiple act calls to see LSTM state update
        dummy_state_obs_2 = np.random.rand(state_s).astype(np.float32)
        action_2 = agent.act(dummy_state_obs_2)
        print(f"agent.act(dummy_state_obs_2) returned action: {action_2}")
        assert agent.agent_lstm_h_c is not None, "LSTM hidden state not updated after act."
        print("LSTM hidden state seems to be updating after act.")

        # Test step method (fill up state_history and potentially the buffer)
        print("Testing agent.step()...")
        for i in range(seq_len + test_batch_size + 1): # Fill history and add enough for a batch
            prev_obs = np.random.rand(state_s).astype(np.float32)
            act_taken = agent.act(prev_obs) # Action is 1-based
            # print(f"  Action taken for step {i}: {act_taken}")
            # print(f"  State history before step: {len(agent.state_history)}")
            # if agent.agent_lstm_h_c:
            #     print(f"  LSTM h shape: {agent.agent_lstm_h_c[0].shape}")

            next_obs = np.random.rand(state_s).astype(np.float32)
            reward_val = np.random.rand()
            done_val = (i == seq_len + test_batch_size) # Make last step done

            agent.step(prev_obs, act_taken, reward_val, next_obs, done_val)
            # print(f"  State history after step: {len(agent.state_history)}")
            # if done_val:
            #    assert agent.agent_lstm_h_c is None, "LSTM state not reset after done."
            #    print("LSTM state correctly reset after done.")
            #    agent.reset_episode_states() # Reset for next "fake" sequence

        print(f"Finished agent.step() loop. Memory size: {len(agent.memory)}")

        # Test learn method if buffer has enough samples
        if len(agent.memory) >= agent.batch_size:
            print("Buffer has enough samples, testing learn()...")
            # Ensure agent is in training mode for learn step
            agent.policy_net.train()
            agent.target_net.train()
            experiences = agent.memory.sample(agent.batch_size)
            agent.learn(experiences)
            print("agent.learn(experiences) called successfully.")
        else:
            print(f"Skipping learn() test, not enough samples in memory. Need {agent.batch_size}, have {len(agent.memory)}.")

        print("D3QNAgent basic tests completed successfully!")

    except Exception as e:
        print(f"An error occurred during D3QNAgent tests: {e}")
        import traceback
        traceback.print_exc()