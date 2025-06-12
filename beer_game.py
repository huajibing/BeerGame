import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from env import Env, EnvExtended # MODIFIED: Import EnvExtended
import os
import argparse
from dqn import DQNAgent
from ppo import PPOAgent, StateNormalizer
from d3qn import D3QNAgent

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Create directories for models and results if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Supply Chain Strategies for non-agent firms
class SupplyChainStrategy:
    """Base class for supply chain strategies"""
    def __init__(self, max_order=20):
        self.max_order = max_order
    
    def select_action(self, state):
        """Select an order quantity based on the current state"""
        raise NotImplementedError("Subclasses must implement this method")

class RandomStrategy(SupplyChainStrategy):
    """Random ordering strategy"""
    def select_action(self, state):
        return np.random.randint(1, self.max_order + 1)

class BaseStockStrategy(SupplyChainStrategy):
    """Base-stock policy: order to reach target inventory level"""
    def __init__(self, target_level, max_order=20):
        super().__init__(max_order)
        self.target_level = target_level
    
    def select_action(self, state):
        current_inventory = state[2]  # Assuming inventory is the third element in state
        order_quantity = max(1, min(self.max_order, int(self.target_level - current_inventory)))
        return order_quantity

class OrderUpToStrategy(SupplyChainStrategy):
    """Order-up-to policy: order when inventory drops below threshold"""
    def __init__(self, threshold, target_level, max_order=20):
        super().__init__(max_order)
        self.threshold = threshold
        self.target_level = target_level
    7.
    def select_action(self, state):
        current_inventory = state[2]  # Assuming inventory is the third element in state
        if current_inventory < self.threshold:
            order_quantity = max(1, min(self.max_order, int(self.target_level - current_inventory)))
        else:
            order_quantity = 1  # Minimum order
        return order_quantity

# Generic training function
def train_agent(
    agent_type,
    env,
    agent_index,
    other_strategies,
    state_size, # MODIFIED: Added state_size parameter
    num_episodes=1000,
    max_steps=100,
    checkpoint_interval=100,
    **agent_kwargs
):
    """Train a reinforcement learning agent for the specified firm"""
    
    # Initialize the agent based on the specified type
    # state_size = 3  # MODIFIED: Removed hardcoded state_size
    action_size = 20  # Max order quantity (assuming this remains constant)
    
    if agent_type.lower() == "dqn":
        agent = DQNAgent(state_size, action_size, **agent_kwargs)
    elif agent_type.lower() == "ppo":
        # PPO默认使用更大的更新频率
        if 'update_frequency' not in agent_kwargs:
            agent_kwargs['update_frequency'] = 1024  # 默认收集1024步经验后更新
        agent = PPOAgent(state_size, action_size, **agent_kwargs)
    elif agent_type.lower() == "d3qn":
        agent = D3QNAgent(state_size, action_size, **agent_kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Lists to track progress
    scores = []
    avg_scores = []
    episode_rewards = []  # 每个episode的奖励
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        if hasattr(agent, 'reset_episode_states'): # For D3QN with LSTM
            agent.reset_episode_states()
        episode_reward = 0
        
        for step in range(max_steps):
            # Prepare actions for all firms
            actions = np.zeros((env.num_firms, 1))
            
            # Get action for the agent firm
            agent_state = state[agent_index]
            agent_action = agent.act(agent_state)
            actions[agent_index] = agent_action
            
            # Get actions for other firms based on their strategies
            for i in range(env.num_firms):
                if i != agent_index:
                    actions[i] = other_strategies[i].select_action(state[i])
            
            # Take step in environment
            next_state, rewards, done = env.step(actions)
            agent_reward = rewards[agent_index][0]  # Get the scalar reward
            
            # Store the experience based on agent type
            if agent_type.lower() == "dqn":
                agent.step(state[agent_index], agent_action, agent_reward, next_state[agent_index], done)
            elif agent_type.lower() == "ppo":
                agent.store_experience(state[agent_index], agent_action, agent_reward, next_state[agent_index], done)
                
                # 检查是否应该更新PPO（按累积步数触发）
                if agent.should_update():
                    agent.update()
            elif agent_type.lower() == "d3qn":
                agent.step(state[agent_index], agent_action, agent_reward, next_state[agent_index], done)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += agent_reward
            
            if done:
                break
        
        # 记录这个episode的总奖励
        episode_rewards.append(episode_reward)
        
        # Track scores (使用窗口平均)
        scores.append(episode_reward)
        if len(scores) > 100:
            scores = scores[-100:]  # 保持最近100个episode的分数
        avg_score = np.mean(scores)
        avg_scores.append(avg_score)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Score: {episode_reward:.2f}, Avg Score: {avg_score:.2f}")
            # Print additional info based on agent type
            if agent_type.lower() == "dqn":
                print(f"Epsilon: {agent.epsilon:.2f}")
            elif agent_type.lower() == "ppo":
                print(f"Steps since update: {agent.steps_since_update}/{agent.update_frequency}")
        
        # Save checkpoint
        if episode % checkpoint_interval == 0:
            model_path = os.path.join("models", f"{agent_type}_agent_firm{agent_index}_episode{episode}.pth")
            if agent_type.lower() == "dqn":
                torch.save(agent.policy_net.state_dict(), model_path)
            elif agent_type.lower() == "ppo":
                torch.save({
                    'network': agent.network.state_dict(),
                    'state_normalizer_mean': agent.state_normalizer.mean if agent.normalize_states else None,
                    'state_normalizer_std': agent.state_normalizer.std if agent.normalize_states else None,
                }, model_path)
    
    # 确保最后的经验被用于更新（如果有足够的数据）
    if agent_type.lower() == "ppo" and len(agent.states) >= agent.batch_size:
        agent.update()
    
    # Save final model
    model_path = os.path.join("models", f"{agent_type}_agent_firm{agent_index}_final.pth")
    if agent_type.lower() == "dqn":
        torch.save(agent.policy_net.state_dict(), model_path)
    elif agent_type.lower() == "ppo":
        torch.save({
            'network': agent.network.state_dict(),
            'state_normalizer_mean': agent.state_normalizer.mean if agent.normalize_states else None,
            'state_normalizer_std': agent.state_normalizer.std if agent.normalize_states else None,
        }, model_path)
    elif agent_type.lower() == "d3qn":
        torch.save(agent.policy_net.state_dict(), model_path)
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.3, label='Episode Rewards')
    plt.plot(avg_scores, label='Average Score (100 ep window)')
    plt.title(f"{agent_type.upper()} Training for Firm {agent_index}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(os.path.join("results", f"{agent_type}_agent_firm{agent_index}_training.png"))
    plt.close()
    
    return agent

# Evaluation function
def evaluate_agent(
    agent_type,
    env,
    agent,
    agent_index,
    other_strategies,
    num_episodes=100,
    max_steps=100
):
    """Evaluate a trained agent's performance"""
    
    # Lists to track metrics
    episode_rewards = []
    episode_inventories = []
    episode_orders = []
    episode_demands = []
    episode_satisfied_demands = []
    
    # Evaluation loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_inventory = []
        episode_order = []
        episode_demand = []
        episode_satisfied_demand = []
        
        for step in range(max_steps):
            # Prepare actions for all firms
            actions = np.zeros((env.num_firms, 1))
            
            # Get action for the agent firm (in evaluation mode)
            agent_state = state[agent_index]
            agent_action = agent.act(agent_state, eval_mode=True)
            actions[agent_index] = agent_action
            
            # Get actions for other firms based on their strategies
            for i in range(env.num_firms):
                if i != agent_index:
                    actions[i] = other_strategies[i].select_action(state[i])
            
            # Take step in environment
            next_state, rewards, done = env.step(actions)
            agent_reward = rewards[agent_index][0]  # Get the scalar reward
            
            # Update state and accumulate reward
            state = next_state
            total_reward += agent_reward
            
            # Track metrics
            episode_inventory.append(env.inventory[agent_index][0])
            episode_order.append(actions[agent_index][0])
            if hasattr(env, 'demand'):
                episode_demand.append(env.demand[agent_index][0])
            episode_satisfied_demand.append(env.satisfied_demand[agent_index][0])
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(total_reward)
        episode_inventories.append(episode_inventory)
        episode_orders.append(episode_order)
        episode_demands.append(episode_demand)
        episode_satisfied_demands.append(episode_satisfied_demand)
    
    # Calculate average metrics
    avg_reward = np.mean(episode_rewards)
    avg_inventory = np.mean([np.mean(inv) for inv in episode_inventories])
    avg_order = np.mean([np.mean(ord) for ord in episode_orders])
    
    print(f"Evaluation Results for {agent_type.upper()} Agent in Firm {agent_index}:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Inventory: {avg_inventory:.2f}")
    print(f"Average Order: {avg_order:.2f}")
    
    # Plot metrics
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title(f"Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot inventory over time (averaged across episodes)
    plt.subplot(2, 2, 2)
    avg_inventory_over_time = np.mean(episode_inventories, axis=0)
    plt.plot(avg_inventory_over_time)
    plt.title(f"Average Inventory over Time")
    plt.xlabel("Step")
    plt.ylabel("Inventory")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot orders over time (averaged across episodes)
    plt.subplot(2, 2, 3)
    avg_order_over_time = np.mean(episode_orders, axis=0)
    plt.plot(avg_order_over_time)
    plt.title(f"Average Order over Time")
    plt.xlabel("Step")
    plt.ylabel("Order Quantity")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot demand vs satisfied demand over time (averaged across episodes)
    if episode_demands:
        plt.subplot(2, 2, 4)
        avg_demand_over_time = np.mean(episode_demands, axis=0)
        avg_satisfied_demand_over_time = np.mean(episode_satisfied_demands, axis=0)
        plt.plot(avg_demand_over_time, label="Demand")
        plt.plot(avg_satisfied_demand_over_time, label="Satisfied Demand")
        plt.title(f"Demand vs Satisfied Demand over Time")
        plt.xlabel("Step")
        plt.ylabel("Quantity")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"{agent_type}_agent_firm{agent_index}_evaluation.png"))
    plt.close()
    
    # 添加奖励分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(episode_rewards, bins=20, alpha=0.7)
    plt.axvline(avg_reward, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_reward:.2f}')
    plt.title(f"Reward Distribution over {num_episodes} Episodes")
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join("results", f"{agent_type}_agent_firm{agent_index}_reward_distribution.png"))
    plt.close()
    
    return {
        'rewards': episode_rewards,
        'inventories': episode_inventories,
        'orders': episode_orders,
        'demands': episode_demands,
        'satisfied_demands': episode_satisfied_demands,
    }

# Compare with baseline strategies
def compare_strategies(
    env,
    target_firm_index,
    state_size,
    args, # MODIFIED: Added args to pass command-line arguments
    agent_types=None,
    other_firm_strategy="random",
    num_episodes=100,
    max_steps=100
):
    """Compare different strategies for a specific position in the supply chain"""
    
    if agent_types is None:
        agent_types = ["dqn", "ppo", "d3qn"]
    
    # Define strategies to compare
    strategies = {
        "Random": RandomStrategy(),
        "BaseStock": BaseStockStrategy(target_level=20), # Assuming state[2] is still inventory for base Env
        "OrderUpTo": OrderUpToStrategy(threshold=10, target_level=20), # Assuming state[2] is still inventory for base Env
    }
    
    # Try to load trained agents if available
    # state_size = 3  # MODIFIED: Removed hardcoded state_size
    action_size = 20 # Max order quantity (assuming this remains constant)
    
    for agent_type in agent_types:
        try:
            model_path = os.path.join("models", f"{agent_type}_agent_firm{target_firm_index}_final.pth")
            
            if agent_type.lower() == "dqn":
                agent = DQNAgent(state_size, action_size)
                agent.policy_net.load_state_dict(torch.load(model_path))
                strategies["DQN"] = agent
            elif agent_type.lower() == "ppo":
                # 为PPO创建带归一化器的代理
                agent = PPOAgent(state_size, action_size, normalize_states=True)
                
                # 加载模型和归一化器参数
                checkpoint = torch.load(model_path)
                agent.network.load_state_dict(checkpoint['network'])
                
                # 如果保存了归一化器数据，加载它
                if 'state_normalizer_mean' in checkpoint and checkpoint['state_normalizer_mean'] is not None:
                    agent.state_normalizer.mean = checkpoint['state_normalizer_mean']
                    agent.state_normalizer.std = checkpoint['state_normalizer_std']
                    agent.state_normalizer.n = 1000  # 设置一个合理的样本数量
                
                strategies["PPO"] = agent
            elif agent_type.lower() == "d3qn":
                # When loading D3QN, ensure architectural params match those used for training
                # These are now passed via 'args'
                d3qn_load_kwargs = {
                    'noisy_std_init': args.d3qn_noisy_std,
                    'sequence_length': args.d3qn_seq_len,
                    'lstm_hidden_size': args.d3qn_lstm_hidden,
                    'num_atoms': args.d3qn_num_atoms,
                    'v_min': args.d3qn_v_min,
                    'v_max': args.d3qn_v_max,
                    # Other necessary D3QNAgent params if they are not architectural (e.g. learning_rate is not needed for eval)
                    # For evaluation, only architectural parameters are strictly necessary.
                }
                agent = D3QNAgent(state_size, action_size, **d3qn_load_kwargs)
                agent.policy_net.load_state_dict(torch.load(model_path))
                strategies["D3QN"] = agent
                
        except Exception as e:
            print(f"Could not load trained {agent_type.upper()} agent: {e}. Skipping comparison.")
    
    # Define strategy for other firms
    other_strategies = []
    for i in range(env.num_firms):
        if i != target_firm_index:
            if other_firm_strategy == "random":
                other_strategies.append(RandomStrategy())
            elif other_firm_strategy == "basestock":
                other_strategies.append(BaseStockStrategy(target_level=20))
            elif other_firm_strategy == "orderupto":
                other_strategies.append(OrderUpToStrategy(threshold=10, target_level=20))
            else:
                other_strategies.append(RandomStrategy())
    
    # Compare strategies
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"Evaluating {strategy_name} strategy at position {target_firm_index}...")
        
        # Track metrics
        strategy_rewards = []
        strategy_inventories = []
        strategy_orders = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if hasattr(strategy, 'reset_episode_states') and isinstance(strategy, D3QNAgent):
                strategy.reset_episode_states()
            episode_reward = 0
            episode_inventory = []
            episode_order = []
            
            for step in range(max_steps):
                # Prepare actions for all firms
                actions = np.zeros((env.num_firms, 1))
                
                # Get action for the target firm based on the strategy
                if strategy_name in ["DQN", "PPO", "D3QN"]:
                    actions[target_firm_index] = strategy.act(state[target_firm_index], eval_mode=True)
                else:
                    actions[target_firm_index] = strategy.select_action(state[target_firm_index])
                
                # Get actions for other firms
                other_idx = 0
                for i in range(env.num_firms):
                    if i != target_firm_index:
                        actions[i] = other_strategies[other_idx].select_action(state[i])
                        other_idx += 1
                
                # Take step in environment
                next_state, rewards, done = env.step(actions)
                target_reward = rewards[target_firm_index][0]
                episode_reward += target_reward
                
                # Track metrics
                episode_inventory.append(env.inventory[target_firm_index][0])
                episode_order.append(actions[target_firm_index][0])
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Store episode metrics
            strategy_rewards.append(episode_reward)
            strategy_inventories.append(np.mean(episode_inventory))
            strategy_orders.append(np.mean(episode_order))
        
        # Calculate average metrics
        avg_reward = np.mean(strategy_rewards)
        avg_inventory = np.mean(strategy_inventories)
        avg_order = np.mean(strategy_orders)
        
        results[strategy_name] = {
            'rewards': strategy_rewards,
            'avg_reward': avg_reward,
            'avg_inventory': avg_inventory,
            'avg_order': avg_order
        }
        
        print(f"{strategy_name} - Avg Reward: {avg_reward:.2f}, Avg Inventory: {avg_inventory:.2f}, Avg Order: {avg_order:.2f}")
    
    # 1. Complete version of charts (including all strategies)
    plt.figure(figsize=(12, 8))
    
    # Box plot of rewards
    plt.subplot(2, 1, 1)
    reward_data = [results[strategy]['rewards'] for strategy in results.keys()]
    plt.boxplot(reward_data, labels=list(results.keys()))
    plt.title(f'Reward Comparison for Firm {target_firm_index}')
    plt.ylabel('Total Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Bar chart of average rewards with value labels
    plt.subplot(2, 1, 2)
    strategies_list = list(results.keys())
    avg_rewards = [results[strategy]['avg_reward'] for strategy in strategies_list]
    bars = plt.bar(strategies_list, avg_rewards)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom' if height > 0 else 'top')
                
    plt.title(f'Average Reward Comparison for Firm {target_firm_index}')
    plt.ylabel('Average Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"strategy_comparison_firm{target_firm_index}.png"))
    plt.close()
    
    # 2. Exclude Random from charts (for better comparison of other strategies)
    non_random_strategies = [s for s in strategies_list if s != "Random"]
    if len(non_random_strategies) > 1:  # Ensure there are at least two non-Random strategies
        plt.figure(figsize=(12, 8))
        
        # Box plot excluding Random
        plt.subplot(2, 1, 1)
        non_random_data = [results[strategy]['rewards'] for strategy in non_random_strategies]
        plt.boxplot(non_random_data, labels=non_random_strategies)
        plt.title(f'Reward Comparison (Excluding Random) for Firm {target_firm_index}')
        plt.ylabel('Total Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Bar chart excluding Random
        plt.subplot(2, 1, 2)
        non_random_rewards = [results[strategy]['avg_reward'] for strategy in non_random_strategies]
        bars = plt.bar(non_random_strategies, non_random_rewards)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}',
                     ha='center', va='bottom' if height > 0 else 'top')
                    
        plt.title(f'Average Reward Comparison (Excluding Random) for Firm {target_firm_index}')
        plt.ylabel('Average Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join("results", f"strategy_comparison_no_random_firm{target_firm_index}.png"))
        plt.close()
    
    # 3. Relative performance chart (with Random as baseline)
    if "Random" in results:
        random_reward = results["Random"]["avg_reward"]
        plt.figure(figsize=(10, 6))
        
        # Calculate improvement percentage relative to Random
        relative_strategies = [s for s in strategies_list if s != "Random"]
        improvement_percentages = [((results[s]['avg_reward'] - random_reward) / abs(random_reward)) * 100 
                                  for s in relative_strategies]
        
        bars = plt.bar(relative_strategies, improvement_percentages)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom' if height > 0 else 'top')
                    
        plt.title(f'Performance Improvement Compared to Random Strategy (Firm {target_firm_index})')
        plt.ylabel('Improvement Percentage (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join("results", f"strategy_improvement_firm{target_firm_index}.png"))
        plt.close()
    
    # 4. Inventory and order comparison charts
    plt.figure(figsize=(15, 6))
    
    # Average inventory comparison
    plt.subplot(1, 3, 1)
    avg_inventories = [results[strategy]['avg_inventory'] for strategy in strategies_list]
    bars = plt.bar(strategies_list, avg_inventories)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')
    
    plt.title(f'Average Inventory (Firm {target_firm_index})')
    plt.ylabel('Average Inventory')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Average order comparison
    plt.subplot(1, 3, 2)
    avg_orders = [results[strategy]['avg_order'] for strategy in strategies_list]
    bars = plt.bar(strategies_list, avg_orders)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')
    
    plt.title(f'Average Order (Firm {target_firm_index})')
    plt.ylabel('Average Order')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Average reward vs. inventory/order scatter plot
    plt.subplot(1, 3, 3)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    for i, strategy in enumerate(strategies_list):
        plt.scatter(results[strategy]['avg_inventory'], 
                   results[strategy]['avg_order'], 
                   s=100, 
                   color=colors[i % len(colors)], 
                   label=f"{strategy} (R:{results[strategy]['avg_reward']:.1f})")
        
    plt.title(f'Inventory vs Order (Firm {target_firm_index})')
    plt.xlabel('Average Inventory')
    plt.ylabel('Average Order')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"strategy_metrics_comparison_firm{target_firm_index}.png"))
    plt.close()
    
    return results

# Main function with support for multiple agent types
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Beer Game Reinforcement Learning Framework')
    parser.add_argument('--agents', type=str, nargs='+', default=['dqn', 'ppo', 'd3qn'], 
                      choices=['dqn', 'ppo', 'd3qn'],
                      help='RL algorithms to use (default: dqn ppo d3qn)')
    parser.add_argument('--firm', type=int, default=1, choices=[0, 1, 2],
                       help='Firm position (0=retailer, 1=wholesaler, 2=factory, default: 1)')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of training episodes (default: 2000)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Maximum steps per episode (default: 100)')
    parser.add_argument('--other-strategy', type=str, default='random', 
                       choices=['random', 'basestock', 'orderupto'],
                       help='Strategy for other firms (default: random)')
    # 新增PPO特定参数
    parser.add_argument('--ppo-update-freq', type=int, default=1024,
                       help='PPO update frequency in steps (default: 1024)')
    parser.add_argument('--ppo-lr', type=float, default=3e-4,
                       help='PPO learning rate (default: 3e-4)')
    parser.add_argument('--ppo-batch', type=int, default=64,
                       help='PPO batch size (default: 64)')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                       help='PPO optimization epochs per update (default: 4)')
    parser.add_argument('--ppo-entropy', type=float, default=1e-3,
                       help='PPO entropy coefficient (default: 1e-3)')
    parser.add_argument('--d3qn-tau', type=float, default=0.005,
                      help='D3QN soft update parameter (default: 0.005)')
    parser.add_argument('--d3qn-lr', type=float, default=0.001,
                      help='D3QN learning rate (default: 0.001)')
    # D3QN specific arguments (including C51, LSTM, Noisy, PER)
    parser.add_argument('--d3qn_gamma', type=float, default=0.99, help='D3QN discount factor gamma (default: 0.99)')
    parser.add_argument('--d3qn_buffer_size', type=int, default=10000, help='D3QN replay buffer size (default: 10000)')
    parser.add_argument('--d3qn_batch_size', type=int, default=64, help='D3QN batch size (default: 64)')
    parser.add_argument('--d3qn_update_every', type=int, default=4, help='D3QN update frequency (default: 4)')
    parser.add_argument('--d3qn_prioritized', action='store_true', help='Enable prioritized experience replay for D3QN')
    parser.add_argument('--d3qn_alpha_per', type=float, default=0.6, help='Alpha for PER in D3QN (default: 0.6)')
    parser.add_argument('--d3qn_beta_per_start', type=float, default=0.4, help='Beta start for PER in D3QN (default: 0.4)')
    parser.add_argument('--d3qn_beta_per_end', type=float, default=1.0, help='Beta end for PER in D3QN (default: 1.0)')
    parser.add_argument('--d3qn_beta_per_frames', type=int, default=100000, help='Beta annealing frames for PER in D3QN (default: 100000)')
    parser.add_argument('--d3qn_noisy_std', type=float, default=0.1, help='Initial std for NoisyLinear layers in D3QN')
    parser.add_argument('--d3qn_seq_len', type=int, default=8, help='Sequence length for LSTM in D3QN')
    parser.add_argument('--d3qn_lstm_hidden', type=int, default=128, help='LSTM hidden size for D3QN')
    parser.add_argument('--d3qn_num_atoms', type=int, default=51, help='Number of atoms for C51 distributional RL in D3QN')
    parser.add_argument('--d3qn_v_min', type=float, default=-10.0, help='Minimum value of support for C51 in D3QN')
    parser.add_argument('--d3qn_v_max', type=float, default=10.0, help='Maximum value of support for C51 in D3QN')

    # MODIFIED: Add new arguments for environment type and its parameters
    parser.add_argument('--env_type', type=str, default='original', choices=['original', 'extended'],
                        help='Type of environment to use (default: original)')
    parser.add_argument('--lead_time', type=int, default=2,
                        help='Lead time for EnvExtended (default: 2)')
    parser.add_argument('--history_length', type=int, default=3,
                        help='History length for EnvExtended (default: 3)')

    # The D3QN specific arguments (noisy_std, seq_len, lstm_hidden, num_atoms, v_min, v_max)
    # were already defined above. This second block is the duplicate.
    # I am removing this duplicate block.
    # parser.add_argument('--d3qn_noisy_std', type=float, default=0.1, help='Initial std for NoisyLinear layers in D3QN')
    # parser.add_argument('--d3qn_seq_len', type=int, default=8, help='Sequence length for LSTM in D3QN')
    # parser.add_argument('--d3qn_lstm_hidden', type=int, default=128, help='LSTM hidden size for D3QN')
    # parser.add_argument('--d3qn_num_atoms', type=int, default=51, help='Number of atoms for C51 distributional RL in D3QN')
    # parser.add_argument('--d3qn_v_min', type=float, default=-10.0, help='Minimum value of support for C51 in D3QN')
    # parser.add_argument('--d3qn_v_max', type=float, default=10.0, help='Maximum value of support for C51 in D3QN')
    # parser.add_argument('--d3qn_alpha_per', type=float, default=0.6, help='Alpha for PER in D3QN')
    # parser.add_argument('--d3qn_beta_per_start', type=float, default=0.4, help='Beta start for PER in D3QN')

    args = parser.parse_args()
    
    # Initialize environment with parameters
    num_firms = 3
    p = [10, 9, 8]  # Price list
    h = 0.5  # Inventory holding cost
    c = 2  # Lost sales cost
    initial_inventory = 100
    poisson_lambda = 10
    max_steps = args.steps

    # MODIFIED: Environment Instantiation
    if args.env_type == 'extended':
        env = EnvExtended(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps,
                          lead_time=args.lead_time, history_length=args.history_length)
        print(f"Using Extended Environment with lead_time={args.lead_time}, history_length={args.history_length}")
    else:
        env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)
        print("Using Original Environment")

    # MODIFIED: Dynamic state_size determination
    state_size = env.state_size
    print(f"Determined state_size: {state_size}")
    
    # Set maximum order quantity
    max_order = 20 # Assuming action space (max order) remains the same
    
    # Set target firm position (from command line argument)
    target_firm_index = args.firm
    
    # Set strategies for other firms
    other_strategies = []
    for i in range(num_firms):
        if i != target_firm_index:
            if args.other_strategy == "random":
                other_strategies.append(RandomStrategy(max_order))
            elif args.other_strategy == "basestock":
                other_strategies.append(BaseStockStrategy(target_level=20, max_order=max_order))
            elif args.other_strategy == "orderupto":
                other_strategies.append(OrderUpToStrategy(threshold=10, target_level=20, max_order=max_order))
        else:
            other_strategies.append(None)  # Placeholder for the agent
    
    # Train and evaluate each requested agent type
    trained_agents = {}
    
    for agent_type in args.agents:
        print(f"\nTraining {agent_type.upper()} agent for Firm {target_firm_index}...")
        
        # 准备特定于代理类型的参数
        agent_kwargs = {}
        
        if agent_type.lower() == "ppo":
            agent_kwargs = {
                'learning_rate': args.ppo_lr,
                'batch_size': args.ppo_batch,
                'epochs': args.ppo_epochs,
                'entropy_coef': args.ppo_entropy,
                'update_frequency': args.ppo_update_freq,
                'normalize_states': True  # 总是启用状态归一化
            }
        elif agent_type.lower() == "d3qn":
            agent_kwargs = {
                'learning_rate': args.d3qn_lr,
                'tau': args.d3qn_tau,
                'gamma': args.d3qn_gamma,
                'buffer_size': args.d3qn_buffer_size,
                'batch_size': args.d3qn_batch_size,
                'update_every': args.d3qn_update_every,
                'prioritized_replay': args.d3qn_prioritized,
                'alpha_per': args.d3qn_alpha_per,
                'beta_per_start': args.d3qn_beta_per_start,
                'beta_per_end': args.d3qn_beta_per_end,
                'beta_per_frames': args.d3qn_beta_per_frames,
                'noisy_std_init': args.d3qn_noisy_std,
                'sequence_length': args.d3qn_seq_len,
                'lstm_hidden_size': args.d3qn_lstm_hidden,
                'num_atoms': args.d3qn_num_atoms,
                'v_min': args.d3qn_v_min,
                'v_max': args.d3qn_v_max,
            }
        
        # Train agent
        agent = train_agent(
            agent_type=agent_type,
            env=env,
            agent_index=target_firm_index,
            other_strategies=other_strategies,
            state_size=state_size, # MODIFIED: Pass state_size
            num_episodes=args.episodes,
            max_steps=max_steps,
            checkpoint_interval=100,
            **agent_kwargs
        )
        
        trained_agents[agent_type] = agent
        
        # Evaluate agent
        print(f"\nEvaluating {agent_type.upper()} agent for Firm {target_firm_index}...")
        evaluate_agent(
            agent_type=agent_type,
            env=env,
            agent=agent,
            agent_index=target_firm_index,
            other_strategies=other_strategies,
            num_episodes=100,
            max_steps=max_steps
        )
    
    # Compare strategies including all trained agents
    print("\nComparing strategies...")
    compare_strategies(
        env=env,
        target_firm_index=target_firm_index,
        state_size=state_size,
        args=args, # MODIFIED: Pass args
        agent_types=args.agents,
        other_firm_strategy=args.other_strategy,
        num_episodes=100,
        max_steps=max_steps
    )

if __name__ == "__main__":
    main()