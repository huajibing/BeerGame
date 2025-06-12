import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# 添加状态归一化器
class StateNormalizer:
    def __init__(self, epsilon=1e-4):
        self.mean = None
        self.std = None
        self.epsilon = epsilon  # 防止除零错误
        self.n = 0
    
    def update(self, states):
        """更新均值和标准差的滚动估计"""
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        
        batch_size = states.shape[0]
        
        # 首次更新
        if self.mean is None:
            self.mean = np.mean(states, axis=0)
            self.std = np.std(states, axis=0)
            self.n = batch_size
            return
        
        # 增量更新均值和标准差
        new_n = self.n + batch_size
        new_mean = (self.mean * self.n + np.sum(states, axis=0)) / new_n
        
        # 增量更新标准差 (使用Welford算法的变种)
        old_s = self.std ** 2 * self.n  # 旧的平方和
        batch_mean = np.mean(states, axis=0)
        batch_s = np.var(states, axis=0) * batch_size  # 批次内的平方和
        
        # 组合两个估计并调整均值的变化
        mean_correction = (batch_mean - self.mean) ** 2 * self.n * batch_size / new_n
        new_s = old_s + batch_s + mean_correction
        
        self.std = np.sqrt(new_s / new_n)
        self.mean = new_mean
        self.n = new_n
    
    def normalize(self, state):
        """归一化状态"""
        if self.mean is None:
            return state
        
        if isinstance(state, torch.Tensor):
            mean = torch.FloatTensor(self.mean).to(state.device)
            std = torch.FloatTensor(self.std).to(state.device)
            return (state - mean) / (std + self.epsilon)
        else:
            return (state - self.mean) / (self.std + self.epsilon)
    
    def normalize_batch(self, states):
        """归一化一批状态"""
        if self.mean is None:
            return states
        
        if isinstance(states, torch.Tensor):
            mean = torch.FloatTensor(self.mean).to(states.device)
            std = torch.FloatTensor(self.std).to(states.device)
            return (states - mean) / (std + self.epsilon)
        else:
            return (states - self.mean) / (self.std + self.epsilon)

# PPO网络，同时包含策略网络和价值网络
class PPONetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(PPONetwork, self).__init__()
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # 策略网络（Actor）- 输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络（Critic）- 评估状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value
    
    def get_action_probs(self, x):
        shared_features = self.shared(x)
        return self.actor(shared_features)
    
    def get_value(self, x):
        shared_features = self.shared(x)
        return self.critic(shared_features)

# 改进的PPO智能体
class PPOAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        max_order=20,  # 最大订单量
        learning_rate=3e-4,  # 降低学习率以提高稳定性
        gamma=0.99,  # 折扣因子
        epsilon=0.2,  # PPO裁剪参数
        value_coef=0.5,  # 价值损失系数
        entropy_coef=0.005,  # 降低熵系数以减少随机性
        epochs=4,  # 每次更新的优化周期数
        batch_size=64,
        gae_lambda=0.95,  # GAE lambda参数
        update_frequency=1024,  # 新增：每收集多少步经验后更新一次
        normalize_states=True  # 新增：是否归一化状态
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.max_order = max_order
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.update_frequency = update_frequency  # 新增更新频率参数
        self.normalize_states = normalize_states  # 是否归一化状态
        
        # 创建PPO网络
        self.network = PPONetwork(state_size, action_size)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 设置设备（CPU或GPU）
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        # 状态归一化器
        if normalize_states:
            self.state_normalizer = StateNormalizer()
        
        # 跟踪收集的经验步数
        self.steps_since_update = 0
        
        # 是否可以更新的标志
        self.ready_to_update = False

        # 损失列表
        self.policy_losses = []
        self.value_losses = []
    
    def act(self, state, eval_mode=False):
        """使用策略网络选择动作"""
        # 归一化状态（如果启用）
        if self.normalize_states:
            norm_state = self.state_normalizer.normalize(state)
        else:
            norm_state = state
            
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
        self.network.eval()
        
        with torch.no_grad():
            action_probs, value = self.network(state_tensor)
        
        # 在评估模式下，使用最可能的动作
        if eval_mode:
            action = action_probs.argmax().item() + 1  # +1 因为动作从1开始
        else:
            # 从分布中采样
            action_dist = Categorical(action_probs)
            action_idx = action_dist.sample()
            log_prob = action_dist.log_prob(action_idx)
            
            # 存储对数概率和价值
            self.log_probs.append(log_prob)
            self.values.append(value)
            
            action = action_idx.item() + 1  # +1 因为动作从1开始
        
        self.network.train()
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验用于PPO更新"""
        self.states.append(state)
        self.actions.append(action - 1)  # -1 转换为从0开始的索引
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        # 如果启用状态归一化，更新归一化器
        if self.normalize_states:
            self.state_normalizer.update(np.array([state]))
        
        # 增加步数计数
        self.steps_since_update += 1
        
        # 检查是否应该更新
        if self.steps_since_update >= self.update_frequency:
            self.ready_to_update = True
    
    def should_update(self):
        """检查是否应该进行策略更新"""
        return self.ready_to_update and len(self.states) >= self.batch_size
    
    def update(self):
        """使用PPO算法更新策略和价值网络"""
        if not self.should_update():
            return  # 不需要更新
        
        # 重置标志
        self.ready_to_update = False
        self.steps_since_update = 0
        
        if len(self.states) == 0:
            return  # 没有经验可更新
        
        # 将经验转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).unsqueeze(1).to(self.device)
        old_log_probs = torch.cat(self.log_probs).detach()
        old_values = torch.cat(self.values).detach()
        
        # 归一化状态（如果启用）
        if self.normalize_states:
            states = self.state_normalizer.normalize_batch(states)
            next_states = self.state_normalizer.normalize_batch(next_states)
        
        # 计算最后一个状态的值函数
        with torch.no_grad():
            if self.dones[-1]:
                next_value = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            else:
                if self.normalize_states:
                    normalized_next_state = self.state_normalizer.normalize(self.next_states[-1])
                    next_state_tensor = torch.FloatTensor(normalized_next_state).unsqueeze(0).to(self.device)
                else:
                    next_state_tensor = torch.FloatTensor(self.next_states[-1]).unsqueeze(0).to(self.device)
                
                next_value = self.network.get_value(next_state_tensor).squeeze()
        
        # 使用广义优势估计(GAE)计算回报和优势
        returns, advantages = self._compute_gae(rewards, old_values, dones, next_value)
        
        # 标准化优势（提高稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 优化策略和价值网络
        for _ in range(self.epochs):
            # 创建小批量
            batch_indices = torch.randperm(len(self.states))
            
            for start_idx in range(0, len(self.states), self.batch_size):
                # 获取小批量索引
                batch_idx = batch_indices[start_idx:start_idx + self.batch_size]
                
                # 获取小批量数据
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                
                # 前向传播
                action_probs, values = self.network(batch_states)
                
                # 创建动作分布
                dist = Categorical(action_probs)
                
                # 计算对数概率
                new_log_probs = dist.log_prob(batch_actions)
                
                # 计算熵（用于探索）
                entropy = dist.entropy().mean()
                
                # 计算比率 (policy / old_policy)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算PPO目标函数
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 优化
                self.optimizer.zero_grad()
                total_loss.backward()
                # 梯度裁剪（保留，有助于稳定训练）
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                # 记录损失
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
        
        # 清空经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def _compute_gae(self, rewards, values, dones, next_values):
        """计算广义优势估计(GAE)"""
        advantages = torch.zeros_like(rewards, device=self.device, dtype=torch.float32)
        last_advantage = 0
        
        # 反向遍历时序数据，计算GAE
        for t in range(len(rewards) - 1, -1, -1):
            # 如果当前状态是终止状态，那么没有下一个状态的值
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            # 计算TD误差：r + gamma * V(s+1) - V(s)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # 计算GAE: A(t) = delta + gamma * lambda * A(t+1)
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # 计算回报: G(s) = A(s) + V(s)
        returns = advantages + values
        
        return returns, advantages

    def get_average_losses(self):
        """计算并返回平均损失，然后清空损失列表"""
        if not self.policy_losses or not self.value_losses:
            return 0.0, 0.0

        avg_policy_loss = np.mean(self.policy_losses)
        avg_value_loss = np.mean(self.value_losses)

        self.policy_losses = []
        self.value_losses = []

        return avg_policy_loss, avg_value_loss