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

def simple_moving_average(data, window_size):
    if not data:
        return []
    # Ensure data is a numpy array for consistent handling with np.nan
    data_arr = np.array(data, dtype=float) # Convert to float to allow NaNs
    if window_size <= 0:
        return data_arr.tolist()

    results = np.full_like(data_arr, np.nan) # Initialize with NaNs

    for i in range(len(data_arr)):
        start_index = max(0, i - window_size + 1)
        window_slice = data_arr[start_index : i + 1]
        if window_slice.size > 0 : # Ensure window is not empty after slicing
             results[i] = np.nanmean(window_slice)
        # else, it remains np.nan, which is fine
    return results.tolist()

def plot_training_dashboard(training_history, agent_type, agent_index, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    if not training_history.get('episode_rewards') or not training_history['episode_rewards']:
        print(f"Warning: No episode rewards found for {agent_type} Firm {agent_index}. Skipping dashboard.")
        return

    episodes = np.arange(1, len(training_history['episode_rewards']) + 1)
    ma_window = min(50, len(episodes) // 2 if len(episodes) >= 2 else 1)
    if ma_window == 0: ma_window = 1


    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{agent_type.upper()} Agent Training Dashboard for Firm {agent_index}', fontsize=16)

    # Top-Left: Reward Convergence
    axs[0, 0].plot(episodes, training_history['episode_rewards'], alpha=0.4, label='Episode Rewards')
    if training_history.get('avg_scores') and len(training_history['avg_scores']) == len(episodes):
        axs[0, 0].plot(episodes, training_history['avg_scores'], color='orange', label='Avg Reward (100 ep window)')
    axs[0, 0].set_title('Reward Convergence')
    axs[0, 0].set_xlabel('Episodes')
    axs[0, 0].set_ylabel('Total Reward')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Top-Right: Cost Dynamics
    if training_history.get('holding_costs') and len(training_history['holding_costs']) == len(episodes):
        axs[0, 1].plot(episodes, simple_moving_average(training_history['holding_costs'], ma_window), label=f'Holding Cost ({ma_window}ep MA)')
    if training_history.get('stockout_costs') and len(training_history['stockout_costs']) == len(episodes):
        axs[0, 1].plot(episodes, simple_moving_average(training_history['stockout_costs'], ma_window), label=f'Stockout Cost ({ma_window}ep MA)')
    axs[0, 1].set_title('Cost Dynamics (Moving Average)')
    axs[0, 1].set_xlabel('Episodes')
    axs[0, 1].set_ylabel('Average Cost')
    if training_history.get('holding_costs') or training_history.get('stockout_costs'):
        axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Bottom-Left: Policy Behavior Exploration
    ax_bl = axs[1, 0]
    if training_history.get('avg_order_quantities') and len(training_history['avg_order_quantities']) == len(episodes):
        ax_bl.plot(episodes, simple_moving_average(training_history['avg_order_quantities'], ma_window), label=f'Avg Order Qty ({ma_window}ep MA)', color='tab:blue')
    ax_bl.set_xlabel('Episodes')
    ax_bl.set_ylabel('Avg Order Quantity', color='tab:blue')
    ax_bl.tick_params(axis='y', labelcolor='tab:blue')
    ax_bl.grid(True, axis='x') # Grid only on x for primary

    ax_bl_twin = ax_bl.twinx()
    if training_history.get('inventory_stabilities') and len(training_history['inventory_stabilities']) == len(episodes):
        ax_bl_twin.plot(episodes, simple_moving_average(training_history['inventory_stabilities'], ma_window), label=f'Inventory StdDev ({ma_window}ep MA)', color='tab:red')
    ax_bl_twin.set_ylabel('Inventory Std Dev', color='tab:red')
    ax_bl_twin.tick_params(axis='y', labelcolor='tab:red')

    ax_bl.set_title('Policy Behavior Exploration (MA)')
    lines, labels = ax_bl.get_legend_handles_labels()
    lines2, labels2 = ax_bl_twin.get_legend_handles_labels()
    if lines or lines2:
        ax_bl_twin.legend(lines + lines2, labels + labels2, loc='upper right')

    # Bottom-Right: Algorithm-Specific Metrics
    ax_br = axs[1, 1]
    ax_br_legends_handles = []
    ax_br_legends_labels = []

    primary_y_label_set = False

    def add_to_br_legend(line_obj, label_text):
        if line_obj and label_text: # line_obj is a list of Line2D objects
             ax_br_legends_handles.append(line_obj[0])
             ax_br_legends_labels.append(label_text)

    if agent_type.lower() in ['dqn', 'd3qn']:
        # Epsilon
        data_list = training_history.get('epsilon_values')
        if data_list and len(data_list) == len(episodes) and not np.all(np.isnan(np.array(data_list,dtype=float))):
            line = ax_br.plot(episodes, data_list, color='green')
            add_to_br_legend(line, 'Epsilon')
        # Beta (for D3QN PER)
        if agent_type.lower() == 'd3qn':
            data_list_beta = training_history.get('beta_values')
            if data_list_beta and len(data_list_beta) == len(episodes) and not np.all(np.isnan(np.array(data_list_beta,dtype=float))):
                line = ax_br.plot(episodes, data_list_beta, color='brown', linestyle=':')
                add_to_br_legend(line, 'Beta (PER)')
        ax_br.set_ylabel('Epsilon / Beta')
        primary_y_label_set = True

        # Q-Loss (twin axis)
        data_list_qloss = training_history.get('q_losses')
        if data_list_qloss and len(data_list_qloss) == len(episodes) and not np.all(np.isnan(np.array(data_list_qloss,dtype=float))):
            ax_br_twin_loss = ax_br.twinx()
            line = ax_br_twin_loss.plot(episodes, simple_moving_average(data_list_qloss, ma_window), color='purple', linestyle='--')
            ax_br_twin_loss.set_ylabel(f'Q-Loss ({ma_window}ep MA)', color='purple')
            ax_br_twin_loss.tick_params(axis='y', labelcolor='purple')
            add_to_br_legend(line, f'Q-Loss ({ma_window}ep MA)')

    elif agent_type.lower() == 'ppo':
        # Policy Loss
        data_list_ploss = training_history.get('policy_losses')
        if data_list_ploss and len(data_list_ploss) == len(episodes) and not np.all(np.isnan(np.array(data_list_ploss,dtype=float))):
            line = ax_br.plot(episodes, simple_moving_average(data_list_ploss, ma_window), color='red')
            ax_br.set_ylabel(f'Policy Loss ({ma_window}ep MA)')
            primary_y_label_set = True
            add_to_br_legend(line, f'Policy Loss ({ma_window}ep MA)')

        # Value Loss (twin axis)
        data_list_vloss = training_history.get('value_losses')
        if data_list_vloss and len(data_list_vloss) == len(episodes) and not np.all(np.isnan(np.array(data_list_vloss,dtype=float))):
            ax_br_twin_loss = ax_br.twinx() # Create twin only if there's data
            line = ax_br_twin_loss.plot(episodes, simple_moving_average(data_list_vloss, ma_window), color='blue', linestyle='--')
            ax_br_twin_loss.set_ylabel(f'Value Loss ({ma_window}ep MA)', color='blue')
            ax_br_twin_loss.tick_params(axis='y', labelcolor='blue')
            add_to_br_legend(line, f'Value Loss ({ma_window}ep MA)')
            if not primary_y_label_set:
                 ax_br.set_ylabel(f'Value Loss ({ma_window}ep MA)')

    ax_br.set_title('Algorithm-Specific Metrics')
    ax_br.set_xlabel('Episodes')
    if ax_br_legends_handles:
        ax_br.legend(ax_br_legends_handles, ax_br_legends_labels, loc='best')
    ax_br.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plot_filename = os.path.join(results_dir, f"{agent_type.lower()}_firm{agent_index}_training_dashboard.png")
    plt.savefig(plot_filename)
    print(f"Training dashboard saved to {plot_filename}")
    plt.close(fig)

def plot_performance_comparison(comparison_results, target_firm_index, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)

    if not comparison_results:
        print("No comparison results to plot.")
        return

    strategy_colors = {
        "DQN": "blue", "PPO": "green", "D3QN": "orange",
        "BaseStock": "grey", "OrderUpTo": "silver", "Random": "lightcoral",
        "DQNAgent": "blue", "PPOAgent": "green", "D3QNAgent": "orange", # Aliases
    }
    default_color = "purple"

    strategies_for_overall = list(comparison_results.keys())
    strategies_for_others = [s_name for s_name in comparison_results.keys() if s_name.lower() != "random"]

    if not strategies_for_overall:
        print("No strategies found in comparison_results.")
        return

    # Chart 1: Overall Performance (Box Plot + Bar Plot) - Includes Random
    fig1, axs1 = plt.subplots(1, 2, figsize=(18, 7))
    fig1.suptitle(f'Overall Performance Comparison (Firm {target_firm_index})', fontsize=16)

    reward_data_overall = []
    reward_labels_overall = []
    for s_name in strategies_for_overall:
        s_data = comparison_results.get(s_name, {})
        if isinstance(s_data.get('rewards'), list) and s_data.get('rewards'):
            reward_data_overall.append(s_data['rewards'])
            reward_labels_overall.append(s_name)

    if reward_data_overall:
        axs1[0].boxplot(reward_data_overall, tick_labels=reward_labels_overall, patch_artist=True, # Changed 'labels' to 'tick_labels'
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color='red'))
    else:
        axs1[0].text(0.5, 0.5, 'No reward data for boxplot', horizontalalignment='center', verticalalignment='center', transform=axs1[0].transAxes)
    axs1[0].set_title('Reward Distribution per Episode')
    axs1[0].set_ylabel('Total Reward')
    axs1[0].tick_params(axis='x', rotation=45) # Removed ha='right'
    axs1[0].grid(True, linestyle='--', alpha=0.7)

    avg_rewards_overall = [comparison_results.get(s_name, {}).get('avg_reward', 0) for s_name in strategies_for_overall]
    std_rewards_overall = []
    for s_name in strategies_for_overall:
        s_data = comparison_results.get(s_name, {})
        rewards_list = s_data.get('rewards')
        if isinstance(rewards_list, list) and len(rewards_list) > 1:
            std_rewards_overall.append(np.std(rewards_list))
        elif isinstance(rewards_list, list) and len(rewards_list) == 1:
             std_rewards_overall.append(0)
        else:
            std_rewards_overall.append(0)

    bar_colors_overall = [strategy_colors.get(s_name, default_color) for s_name in strategies_for_overall]

    if strategies_for_overall:
        bars1 = axs1[1].bar(strategies_for_overall, avg_rewards_overall, yerr=std_rewards_overall, capsize=5, color=bar_colors_overall, alpha=0.8)
        axs1[1].set_title('Average Reward (+/- Std.Dev. of episode rewards)')
        axs1[1].set_ylabel('Average Reward')
        axs1[1].tick_params(axis='x', rotation=45) # Removed ha='right'
        axs1[1].grid(True, linestyle='--', alpha=0.7)
        for bar_idx, bar in enumerate(bars1):
            actual_avg_reward = avg_rewards_overall[bar_idx]
            y_offset = 0.02 * axs1[1].get_ylim()[1] # Small offset based on y-axis range
            y_text = actual_avg_reward + y_offset if actual_avg_reward >= 0 else actual_avg_reward - y_offset
            va_text = 'bottom' if actual_avg_reward >= 0 else 'top'
            axs1[1].text(bar.get_x() + bar.get_width()/2.0, y_text, f'{actual_avg_reward:.1f}',
                         va=va_text, ha='center')
    else:
        axs1[1].text(0.5, 0.5, 'No data for average reward plot', horizontalalignment='center', verticalalignment='center', transform=axs1[1].transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(results_dir, f"comparison_overall_performance_firm{target_firm_index}.png"))
    plt.close(fig1)

    if not strategies_for_others:
        print("Skipping further comparison plots as only 'Random' strategy might be available or no non-random strategies found.")
        return

    bar_colors_others = [strategy_colors.get(s_name, default_color) for s_name in strategies_for_others]

    # Chart 2: Cost & Service Level Analysis
    fig2, axs2 = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle(f'Cost & Service Level Analysis (Firm {target_firm_index}, Excluding Random)', fontsize=16)

    avg_holding_costs = [comparison_results.get(s_name, {}).get('avg_holding_cost', 0) for s_name in strategies_for_others]
    avg_stockout_costs = [comparison_results.get(s_name, {}).get('avg_stockout_cost', 0) for s_name in strategies_for_others]

    if strategies_for_others:
        axs2[0].bar(strategies_for_others, avg_holding_costs, label='Avg Holding Cost', color=[strategy_colors.get(s, default_color) for s in strategies_for_others], alpha=0.7)
        axs2[0].bar(strategies_for_others, avg_stockout_costs, bottom=avg_holding_costs, label='Avg Stockout Cost', color=[strategy_colors.get(s, default_color) for s in strategies_for_others], alpha=0.5, hatch='//')
    axs2[0].set_title('Average Cost Structure')
    axs2[0].set_ylabel('Average Cost')
    axs2[0].tick_params(axis='x', rotation=45) # Removed ha='right'
    if strategies_for_others: axs2[0].legend()
    axs2[0].grid(True, linestyle='--', alpha=0.7)

    avg_service_levels = [comparison_results.get(s_name, {}).get('avg_service_level', 0) * 100 for s_name in strategies_for_others]
    if strategies_for_others:
        bars_sl = axs2[1].bar(strategies_for_others, avg_service_levels, color=bar_colors_others)
        max_sl_val = max(avg_service_levels) if avg_service_levels else 0
        axs2[1].set_ylim(0, max(105, max_sl_val + 5 if max_sl_val > 0 else 105))
        for bar_idx, bar in enumerate(bars_sl):
            actual_sl = avg_service_levels[bar_idx]
            y_offset = 0.02 * axs2[1].get_ylim()[1] # Small offset
            y_text = actual_sl + y_offset
            axs2[1].text(bar.get_x() + bar.get_width()/2.0, y_text, f'{actual_sl:.1f}%',
                         va='bottom', ha='center')
    axs2[1].set_title('Average Service Level')
    axs2[1].set_ylabel('Service Level (%)')
    if not strategies_for_others: axs2[1].set_ylim(0,105)
    axs2[1].tick_params(axis='x', rotation=45) # Removed ha='right'
    axs2[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(os.path.join(results_dir, f"comparison_cost_service_firm{target_firm_index}.png"))
    plt.close(fig2)

    # Chart 3: Bullwhip Effect & Stability
    fig3, axs3 = plt.subplots(1, 2, figsize=(18, 7))
    fig3.suptitle(f'Bullwhip Effect & Inventory Stability (Firm {target_firm_index}, Excluding Random)', fontsize=16)

    avg_bullwhip_ratios = [comparison_results.get(s_name, {}).get('avg_bullwhip_ratio', 0) for s_name in strategies_for_others]
    if strategies_for_others:
        bars_bw = axs3[0].bar(strategies_for_others, avg_bullwhip_ratios, color=bar_colors_others)
        for bar_idx, bar in enumerate(bars_bw):
            actual_bw = avg_bullwhip_ratios[bar_idx]
            y_offset = 0.02 * axs3[0].get_ylim()[1] # Small offset
            y_text = actual_bw + y_offset
            axs3[0].text(bar.get_x() + bar.get_width()/2.0, y_text, f'{actual_bw:.2f}',
                         va='bottom', ha='center')
    axs3[0].set_title('Average Bullwhip Ratio (Order StdDev / Demand StdDev)')
    axs3[0].set_ylabel('Bullwhip Ratio')
    axs3[0].tick_params(axis='x', rotation=45) # Removed ha='right'
    axs3[0].grid(True, linestyle='--', alpha=0.7)

    avg_inventory_stabilities = [comparison_results.get(s_name, {}).get('avg_inventory_stability', 0) for s_name in strategies_for_others]
    if strategies_for_others:
        bars_is = axs3[1].bar(strategies_for_others, avg_inventory_stabilities, color=bar_colors_others)
        for bar_idx, bar in enumerate(bars_is):
            actual_is = avg_inventory_stabilities[bar_idx]
            y_offset = 0.02 * axs3[1].get_ylim()[1] # Small offset
            y_text = actual_is + y_offset
            axs3[1].text(bar.get_x() + bar.get_width()/2.0, y_text, f'{actual_is:.2f}',
                         va='bottom', ha='center')
    axs3[1].set_title('Average Inventory Stability (Inventory Std.Dev.)')
    axs3[1].set_ylabel('Inventory Standard Deviation')
    axs3[1].tick_params(axis='x', rotation=45) # Removed ha='right'
    axs3[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.savefig(os.path.join(results_dir, f"comparison_bullwhip_stability_firm{target_firm_index}.png"))
    plt.close(fig3)

    # Chart 4: Typical Behavior Trajectory
    fig4, axs4 = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig4.suptitle(f'Typical Behavior Trajectories (Firm {target_firm_index}, Excluding Random)', fontsize=16)
    max_steps_plot = 0

    for s_name in strategies_for_others:
        strategy_data = comparison_results.get(s_name, {})
        ep_inv_trajectories = strategy_data.get('inventories', [])
        if isinstance(ep_inv_trajectories, list) and all(isinstance(traj, list) for traj in ep_inv_trajectories):
            valid_inv_trajectories = [traj for traj in ep_inv_trajectories if traj]
            if valid_inv_trajectories:
                min_len_inv = min(len(traj) for traj in valid_inv_trajectories)
                if min_len_inv > 0:
                    padded_inv_trajectories = [traj[:min_len_inv] for traj in valid_inv_trajectories if len(traj) >= min_len_inv]
                    if padded_inv_trajectories:
                        avg_inv_trajectory = np.mean(padded_inv_trajectories, axis=0)
                        axs4[0].plot(avg_inv_trajectory, label=s_name, color=strategy_colors.get(s_name, default_color), alpha=0.8)
                        if len(avg_inv_trajectory) > max_steps_plot: max_steps_plot = len(avg_inv_trajectory)

        ep_order_trajectories = strategy_data.get('orders', [])
        if isinstance(ep_order_trajectories, list) and all(isinstance(traj, list) for traj in ep_order_trajectories):
            valid_ord_trajectories = [traj for traj in ep_order_trajectories if traj]
            if valid_ord_trajectories:
                min_len_ord = min(len(traj) for traj in valid_ord_trajectories)
                if min_len_ord > 0:
                    padded_ord_trajectories = [traj[:min_len_ord] for traj in valid_ord_trajectories if len(traj) >= min_len_ord]
                    if padded_ord_trajectories:
                        avg_order_trajectory = np.mean(padded_ord_trajectories, axis=0)
                        axs4[1].plot(avg_order_trajectory, label=s_name, color=strategy_colors.get(s_name, default_color), alpha=0.8)
                        if len(avg_order_trajectory) > max_steps_plot: max_steps_plot = len(avg_order_trajectory)

    axs4[0].set_title('Average Inventory Level Over Time')
    axs4[0].set_ylabel('Average Inventory')
    if any(axs4[0].get_lines()): axs4[0].legend()
    axs4[0].grid(True, linestyle='--', alpha=0.7)

    axs4[1].set_title('Average Order Quantity Over Time')
    axs4[1].set_xlabel('Time Step')
    axs4[1].set_ylabel('Average Order Quantity')
    if any(axs4[1].get_lines()): axs4[1].legend()
    axs4[1].grid(True, linestyle='--', alpha=0.7)

    if max_steps_plot > 0:
        axs4[0].set_xlim(0, max_steps_plot -1 if max_steps_plot > 1 else 1)
        axs4[1].set_xlim(0, max_steps_plot -1 if max_steps_plot > 1 else 1)
    else:
        axs4[0].set_xlim(0, 1)
        axs4[1].set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig4.savefig(os.path.join(results_dir, f"comparison_behavior_trajectory_firm{target_firm_index}.png"))
    plt.close(fig4)

    print(f"Comparison plots saved to {results_dir} for firm {target_firm_index}.")

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

class BaselineAgentWrapper:
    def __init__(self, baseline_strategy_instance, strategy_name="baseline"):
        self.strategy = baseline_strategy_instance
        self.agent_type = strategy_name # For compatibility with evaluate_agent's naming

    def act(self, state, eval_mode=True): # eval_mode is ignored but part of signature
        return self.strategy.select_action(state)

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
    # scores = [] # Will be part of training_history or local to episode loop for avg_score calculation
    # avg_scores = [] # Will be training_history['avg_scores']
    # episode_rewards = []  # Will be training_history['episode_rewards']

    total_steps = 0
    training_history = {
        'episode_rewards': [],
        'avg_scores': [], # Moving avg_scores here
        'total_env_steps': [],
        'holding_costs': [],
        'stockout_costs': [],
        'avg_order_quantities': [],
        'inventory_stabilities': [],
        'epsilon_values': [],
        'beta_values': [], # For D3QN with PER
        'q_losses': [],       # For DQN/D3QN
        'policy_losses': [],  # For PPO
        'value_losses': []    # For PPO
    }
    # Local list for 100-episode average score calculation
    last_100_scores = []
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        current_ep_holding_cost = 0
        current_ep_stockout_cost = 0
        current_ep_orders = []
        current_ep_inventories = []
        
        for step in range(max_steps):
            # Prepare actions for all firms
            actions = np.zeros((env.num_firms, 1))
            
            # Get action for the agent firm
            agent_state = state[agent_index]
            agent_action = agent.act(agent_state)
            actions[agent_index] = agent_action
            
            # Get actions for other firms based on their strategies
            other_strategy_idx = 0
            for i in range(env.num_firms):
                if i != agent_index:
                    actions[i] = other_strategies[other_strategy_idx].select_action(state[i])
                    other_strategy_idx += 1
            
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
            total_steps += 1

            # Collect step-specific data for training history
            agent_inventory = env.inventory[agent_index][0]
            agent_action_value = actions[agent_index][0] # Actual order placed

            agent_demand_faced = 0 # Default if not available
            if hasattr(env, 'demand') and env.demand is not None and agent_index < len(env.demand) and env.demand[agent_index] is not None:
                agent_demand_faced = env.demand[agent_index][0]

            agent_satisfied_single_step = env.satisfied_demand[agent_index][0]

            step_holding_cost = env.h * agent_inventory
            unmet_demand_step = agent_demand_faced - agent_satisfied_single_step
            step_stockout_cost = unmet_demand_step * env.c if unmet_demand_step > 0 else 0

            current_ep_holding_cost += step_holding_cost
            current_ep_stockout_cost += step_stockout_cost
            current_ep_orders.append(agent_action_value)
            current_ep_inventories.append(agent_inventory)
            
            if done:
                break
        
        # Append to training_history
        training_history['episode_rewards'].append(episode_reward)
        training_history['holding_costs'].append(current_ep_holding_cost)
        training_history['stockout_costs'].append(current_ep_stockout_cost)
        training_history['avg_order_quantities'].append(np.mean(current_ep_orders) if current_ep_orders else 0)
        training_history['inventory_stabilities'].append(np.std(current_ep_inventories) if len(current_ep_inventories) > 1 else 0)
        training_history['total_env_steps'].append(total_steps)
        
        # Track scores for 100-episode average
        last_100_scores.append(episode_reward)
        if len(last_100_scores) > 100:
            last_100_scores = last_100_scores[-100:]
        avg_score = np.mean(last_100_scores)
        training_history['avg_scores'].append(avg_score)

        # Algorithm-specific metrics
        if agent_type.lower() in ["dqn", "d3qn"]:
            training_history['epsilon_values'].append(agent.epsilon)
            training_history['q_losses'].append(agent.get_average_q_loss()) # Assumes get_average_q_loss clears its internal list
            if agent_type.lower() == "d3qn" and hasattr(agent, 'prioritized_replay') and agent.prioritized_replay:
                training_history['beta_values'].append(agent.beta)
            else:
                training_history['beta_values'].append(np.nan)
            # Placeholders for PPO metrics
            training_history['policy_losses'].append(np.nan)
            training_history['value_losses'].append(np.nan)
        elif agent_type.lower() == "ppo":
            avg_policy_loss, avg_value_loss = agent.get_average_losses() # Assumes get_average_losses clears its internal lists
            training_history['policy_losses'].append(avg_policy_loss)
            training_history['value_losses'].append(avg_value_loss)
            # Placeholders for DQN/D3QN metrics
            training_history['epsilon_values'].append(np.nan)
            training_history['q_losses'].append(np.nan)
            training_history['beta_values'].append(np.nan)
        
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
    
    # Comment out or remove original plot saving:
    # plt.figure(figsize=(10, 6))
    # plt.plot(training_history['episode_rewards'], alpha=0.3, label='Episode Rewards')
    # plt.plot(training_history['avg_scores'], label='Average Score (100 ep window)')
    # plt.title(f"{agent_type.upper()} Training for Firm {agent_index}")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.legend()
    # plt.savefig(os.path.join("results", f"{agent_type.lower()}_agent_firm{agent_index}_training.png"))
    # plt.close()

    # Call the new dashboard plotting function
    plot_training_dashboard(training_history, agent_type, agent_index, results_dir=os.path.join("results"))
    
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
    
    # Lists to track metrics (renamed and new lists added)
    all_ep_rewards = []  # Renamed from episode_rewards
    all_ep_inventories = []  # Renamed from episode_inventories
    all_ep_agent_orders = []  # Renamed from episode_orders
    all_ep_agent_demands_faced = []  # Renamed from episode_demands
    all_ep_agent_satisfied_demands = []  # Renamed from episode_satisfied_demands
    all_ep_total_holding_costs = []
    all_ep_total_stockout_costs = []
    all_ep_service_levels = []
    all_ep_bullwhip_ratios = []
    all_ep_inventory_stabilities = []
    
    # Evaluation loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        # Per-episode accumulators/lists
        current_ep_inventory_levels = []  # Replaces episode_inventory
        current_ep_agent_orders = []  # Replaces episode_order
        current_ep_agent_demands_faced = []  # Replaces episode_demand
        current_ep_agent_satisfied_demand_values = []  # Replaces episode_satisfied_demand
        current_ep_holding_cost = 0
        current_ep_stockout_cost = 0
        
        for step in range(max_steps):
            # Prepare actions for all firms
            actions = np.zeros((env.num_firms, 1))
            
            # Get action for the agent firm (in evaluation mode)
            agent_state = state[agent_index]
            agent_action = agent.act(agent_state, eval_mode=True)
            actions[agent_index] = agent_action
            
            # Get actions for other firms based on their strategies
            other_strategy_idx = 0
            for i in range(env.num_firms):
                if i != agent_index:
                    actions[i] = other_strategies[other_strategy_idx].select_action(state[i])
                    other_strategy_idx += 1
            
            # Take step in environment
            next_state, rewards, done = env.step(actions)
            agent_reward = rewards[agent_index][0]  # Get the scalar reward
            
            # Update state and accumulate reward
            state = next_state
            total_reward += agent_reward
            
            # Retrieve and calculate step-specific metrics
            agent_inventory = env.inventory[agent_index][0]
            agent_order = actions[agent_index][0]
            agent_demand_faced = 0 # Default if not available
            if hasattr(env, 'demand') and env.demand is not None and agent_index < len(env.demand) and env.demand[agent_index] is not None:
                agent_demand_faced = env.demand[agent_index][0]

            agent_satisfied_single_step = env.satisfied_demand[agent_index][0]

            step_holding_cost = env.h * agent_inventory
            unmet_demand_step = agent_demand_faced - agent_satisfied_single_step
            step_stockout_cost = unmet_demand_step * env.c if unmet_demand_step > 0 else 0

            # Append to per-episode lists
            current_ep_inventory_levels.append(agent_inventory)
            current_ep_agent_orders.append(agent_order)
            current_ep_agent_demands_faced.append(agent_demand_faced)
            current_ep_agent_satisfied_demand_values.append(agent_satisfied_single_step)

            # Accumulate costs
            current_ep_holding_cost += step_holding_cost
            current_ep_stockout_cost += step_stockout_cost
            
            if done:
                break
        
        # Store episode metrics
        all_ep_rewards.append(total_reward) # Appending to renamed list
        all_ep_inventories.append(current_ep_inventory_levels) # Appending new list
        all_ep_agent_orders.append(current_ep_agent_orders) # Appending new list
        all_ep_agent_demands_faced.append(current_ep_agent_demands_faced) # Appending new list
        all_ep_agent_satisfied_demands.append(current_ep_agent_satisfied_demand_values) # Appending new list
        all_ep_total_holding_costs.append(current_ep_holding_cost)
        all_ep_total_stockout_costs.append(current_ep_stockout_cost)

        # Calculate and append per-episode metrics
        ep_total_satisfied = np.sum(current_ep_agent_satisfied_demand_values)
        ep_total_demand = np.sum(current_ep_agent_demands_faced)
        ep_service_level = ep_total_satisfied / ep_total_demand if ep_total_demand > 0 else 1.0
        all_ep_service_levels.append(ep_service_level)

        std_orders = np.std(current_ep_agent_orders) if len(current_ep_agent_orders) > 1 else 0.0
        std_demands = np.std(current_ep_agent_demands_faced) if len(current_ep_agent_demands_faced) > 1 else 0.0
        ep_bullwhip_ratio = std_orders / std_demands if std_demands > 0 else 1.0
        all_ep_bullwhip_ratios.append(ep_bullwhip_ratio)

        ep_inv_stability = np.std(current_ep_inventory_levels) if len(current_ep_inventory_levels) > 1 else 0.0
        all_ep_inventory_stabilities.append(ep_inv_stability)
    
    # Calculate average metrics
    avg_reward = np.mean(all_ep_rewards) if all_ep_rewards else 0.0
    avg_holding_cost = np.mean(all_ep_total_holding_costs) if all_ep_total_holding_costs else 0.0
    avg_stockout_cost = np.mean(all_ep_total_stockout_costs) if all_ep_total_stockout_costs else 0.0
    avg_service_level = np.mean(all_ep_service_levels) if all_ep_service_levels else 0.0
    avg_bullwhip_ratio = np.mean(all_ep_bullwhip_ratios) if all_ep_bullwhip_ratios else 0.0
    avg_inventory_stability = np.mean(all_ep_inventory_stabilities) if all_ep_inventory_stabilities else 0.0

    # Original average calculations (ensure they use the new list names)
    avg_inventory_overall = np.mean([np.mean(inv_list) for inv_list in all_ep_inventories if inv_list]) if any(all_ep_inventories) else 0.0
    avg_order_overall = np.mean([np.mean(order_list) for order_list in all_ep_agent_orders if order_list]) if any(all_ep_agent_orders) else 0.0
    
    print(f"Evaluation Results for {agent_type.upper()} Agent in Firm {agent_index}:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Holding Cost: {avg_holding_cost:.2f}")
    print(f"Average Stockout Cost: {avg_stockout_cost:.2f}")
    print(f"Average Service Level: {avg_service_level:.2%}") # Print as percentage
    print(f"Average Bullwhip Ratio: {avg_bullwhip_ratio:.2f}")
    print(f"Average Inventory Stability (Std Dev): {avg_inventory_stability:.2f}")
    print(f"Overall Average Inventory Level: {avg_inventory_overall:.2f}") # Renamed for clarity
    print(f"Overall Average Order Quantity: {avg_order_overall:.2f}") # Renamed for clarity
    
    # Plot metrics
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(all_ep_rewards) # Use new list name
    plt.title(f"Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot inventory over time (averaged across episodes)
    plt.subplot(2, 2, 2)
    # Ensure list is not empty before averaging
    if all_ep_inventories and all_ep_inventories[0]: # Check if there's at least one episode with steps
        avg_inventory_over_time = np.mean(all_ep_inventories, axis=0)
        plt.plot(avg_inventory_over_time)
    else:
        plt.plot([], label="No inventory data") # Plot empty if no data
    plt.title(f"Average Inventory over Time")
    plt.xlabel("Step")
    plt.ylabel("Inventory")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot orders over time (averaged across episodes)
    plt.subplot(2, 2, 3)
    if all_ep_agent_orders and all_ep_agent_orders[0]: # Check if there's at least one episode with steps
        avg_order_over_time = np.mean(all_ep_agent_orders, axis=0)
        plt.plot(avg_order_over_time)
    else:
        plt.plot([], label="No order data") # Plot empty if no data
    plt.title(f"Average Order over Time")
    plt.xlabel("Step")
    plt.ylabel("Order Quantity")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot demand vs satisfied demand over time (averaged across episodes)
    # Check if all_ep_agent_demands_faced and all_ep_agent_satisfied_demands have data
    if all_ep_agent_demands_faced and all_ep_agent_demands_faced[0] and \
       all_ep_agent_satisfied_demands and all_ep_agent_satisfied_demands[0]:
        plt.subplot(2, 2, 4)
        avg_demand_over_time = np.mean(all_ep_agent_demands_faced, axis=0)
        avg_satisfied_demand_over_time = np.mean(all_ep_agent_satisfied_demands, axis=0)
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
    if all_ep_rewards: # Check if list is not empty
        plt.hist(all_ep_rewards, bins=20, alpha=0.7)
        plt.axvline(avg_reward, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {avg_reward:.2f}')
    else:
        plt.text(0.5, 0.5, "No reward data for histogram", ha='center', va='center')
    plt.title(f"Reward Distribution over {num_episodes} Episodes")
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join("results", f"{agent_type}_agent_firm{agent_index}_reward_distribution.png"))
    plt.close()
    
    return {
        'rewards': all_ep_rewards,
        'inventories': all_ep_inventories,
        'orders': all_ep_agent_orders,
        'demands': all_ep_agent_demands_faced,
        'satisfied_demands_steps': all_ep_agent_satisfied_demands,
        'holding_costs_episode': all_ep_total_holding_costs,
        'stockout_costs_episode': all_ep_total_stockout_costs,
        'service_levels_episode': all_ep_service_levels,
        'bullwhip_ratios_episode': all_ep_bullwhip_ratios,
        'inventory_stabilities_episode': all_ep_inventory_stabilities,
        'avg_reward': avg_reward,
        'avg_holding_cost': avg_holding_cost,
        'avg_stockout_cost': avg_stockout_cost,
        'avg_service_level': avg_service_level,
        'avg_bullwhip_ratio': avg_bullwhip_ratio,
        'avg_inventory_stability': avg_inventory_stability,
        'avg_inventory_level': avg_inventory_overall, # Matches updated print statement
        'avg_order_quantity': avg_order_overall, # Matches updated print statement
    }

# Compare with baseline strategies
def compare_strategies(
    env,
    target_firm_index,
    state_size, # MODIFIED: Added state_size parameter
    agent_types=None,  # List of agent types to include in comparison
    other_firm_strategy="random",
    num_episodes=100,
    max_steps=100
):
    """Compare different strategies for a specific position in the supply chain"""
    
    if agent_types is None:
        agent_types = ["dqn", "ppo", "d3qn"]  # Default to both DQN and PPO
    
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
                # Load with weights_only=False as the checkpoint contains numpy arrays for state_normalizer
                checkpoint = torch.load(model_path, weights_only=False)
                agent.network.load_state_dict(checkpoint['network'])
                
                # 如果保存了归一化器数据，加载它
                if 'state_normalizer_mean' in checkpoint and checkpoint['state_normalizer_mean'] is not None:
                    agent.state_normalizer.mean = checkpoint['state_normalizer_mean']
                    agent.state_normalizer.std = checkpoint['state_normalizer_std']
                    agent.state_normalizer.n = 1000  # 设置一个合理的样本数量
                
                strategies["PPO"] = agent
            elif agent_type.lower() == "d3qn":
                agent = D3QNAgent(state_size, action_size)
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
    evaluation_results_for_comparison = {} # Use a new dict name for clarity

    for strategy_name, strategy_object_or_agent in strategies.items():
        print(f"Evaluating {strategy_name} strategy for Firm {target_firm_index}...")
        
        current_agent_for_eval = None
        current_agent_type_for_eval = strategy_name # Default to strategy name

        if strategy_name in ["DQN", "PPO", "D3QN"]: # RL Agents
            current_agent_for_eval = strategy_object_or_agent
            # agent_type is already part of the loaded agent, but we can use strategy_name
        elif isinstance(strategy_object_or_agent, SupplyChainStrategy): # Baseline strategies
            current_agent_for_eval = BaselineAgentWrapper(strategy_object_or_agent, strategy_name)
            current_agent_type_for_eval = strategy_name # e.g. "Random", "BaseStock"
        else:
            print(f"Warning: Unknown strategy type for {strategy_name}. Skipping.")
            continue
        
        eval_metrics = evaluate_agent(
            agent_type=current_agent_type_for_eval, # e.g. "DQN", "Random"
            env=env,
            agent=current_agent_for_eval,
            agent_index=target_firm_index,
            other_strategies=other_strategies, # This is the list of strategies for other firms
            num_episodes=num_episodes, # Use compare_strategies' num_episodes
            max_steps=max_steps        # Use compare_strategies' max_steps
        )
        
        evaluation_results_for_comparison[strategy_name] = eval_metrics
        
        # Print summary of new metrics for this strategy
        print(f"{strategy_name} Evaluation Summary (Firm {target_firm_index}):")
        # Use .get with a default for potentially missing keys if evaluate_agent structure varies
        avg_reward_val = eval_metrics.get('avg_reward', np.nan)
        avg_holding_cost_val = eval_metrics.get('avg_holding_cost', np.nan)
        avg_stockout_cost_val = eval_metrics.get('avg_stockout_cost', np.nan)
        avg_service_level_val = eval_metrics.get('avg_service_level', np.nan)
        avg_bullwhip_ratio_val = eval_metrics.get('avg_bullwhip_ratio', np.nan)
        avg_inventory_stability_val = eval_metrics.get('avg_inventory_stability', np.nan)
        avg_inventory_level_val = eval_metrics.get('avg_inventory_level', np.nan)
        avg_order_quantity_val = eval_metrics.get('avg_order_quantity', np.nan)

        print(f"  Avg Reward: {avg_reward_val:.2f}")
        print(f"  Avg Holding Cost: {avg_holding_cost_val:.2f}")
        print(f"  Avg Stockout Cost: {avg_stockout_cost_val:.2f}")
        print(f"  Avg Service Level: {avg_service_level_val*100:.1f}%" if not np.isnan(avg_service_level_val) else "N/A")
        print(f"  Avg Bullwhip Ratio: {avg_bullwhip_ratio_val:.2f}")
        print(f"  Avg Inventory Stability (StdDev): {avg_inventory_stability_val:.2f}")
        print(f"  Avg Inventory Level: {avg_inventory_level_val:.2f}")
        print(f"  Avg Order Quantity: {avg_order_quantity_val:.2f}")
        print("-" * 30)

    # Comment out ALL existing plotting code within compare_strategies.
    # This will be replaced by a call to a new function in Step 6.
    # For example:
    # # plt.figure(figsize=(12, 8)) ... plt.close()
    # # plt.figure(figsize=(12, 8)) ... plt.close() (for non-random)
    # # plt.figure(figsize=(10, 6)) ... plt.close() (for relative performance)
    # # plt.figure(figsize=(15, 6)) ... plt.close() (for inventory/order comparison)
    
    return evaluation_results_for_comparison

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
    parser.add_argument('--d3qn-prioritized', action='store_true',
                      help='Enable prioritized experience replay for D3QN')
    # MODIFIED: Add new arguments for environment type and its parameters
    parser.add_argument('--env_type', type=str, default='original', choices=['original', 'extended'],
                        help='Type of environment to use (default: original)')
    parser.add_argument('--lead_time', type=int, default=2,
                        help='Lead time for EnvExtended (default: 2)')
    parser.add_argument('--history_length', type=int, default=3,
                        help='History length for EnvExtended (default: 3)')
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
    # This list is for train_agent, evaluate_agent (direct calls), and compare_strategies.
    # It should contain actual strategy objects for firms other than the target_firm_index.
    other_strategies_list_for_env = []
    for i in range(num_firms):
        if i != target_firm_index:
            if args.other_strategy == "random":
                other_strategies_list_for_env.append(RandomStrategy(max_order))
            elif args.other_strategy == "basestock":
                other_strategies_list_for_env.append(BaseStockStrategy(target_level=20, max_order=max_order))
            elif args.other_strategy == "orderupto":
                other_strategies_list_for_env.append(OrderUpToStrategy(threshold=10, target_level=20, max_order=max_order))
            # Add other baseline strategies here if needed
    
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
                'prioritized_replay': args.d3qn_prioritized
            }
        
        # Train agent
        agent = train_agent(
            agent_type=agent_type,
            env=env,
            agent_index=target_firm_index,
            other_strategies=other_strategies_list_for_env, # Pass the filtered list
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
            other_strategies=other_strategies_list_for_env, # Pass the filtered list
            num_episodes=100,
            max_steps=max_steps
        )

        # >>> START OVERFITTING ANALYSIS BLOCK <<<
        print(f"\n--- Overfitting Analysis for {agent_type.upper()} agent on Firm {target_firm_index} ---")

        # Define parameters for the overfitting check environment
        # Example: Modify Poisson lambda. Keep other base parameters.
        poisson_lambda_overfit = int(poisson_lambda * 1.2) # e.g., 20% increase
        if poisson_lambda_overfit == poisson_lambda: # Ensure it actually changes
            poisson_lambda_overfit = poisson_lambda + 2

        print(f"Original poisson_lambda: {poisson_lambda}, Overfit check poisson_lambda: {poisson_lambda_overfit}")

        # Create a new environment instance with modified parameters
        env_overfit_params = {
            'num_firms': num_firms,
            'p': p,
            'h': h,
            'c': c,
            'initial_inventory': initial_inventory,
            'poisson_lambda': poisson_lambda_overfit, # Modified parameter
            'max_steps': max_steps
        }
        if args.env_type == 'extended':
            env_overfit_params.update({
                'lead_time': args.lead_time,
                'history_length': args.history_length
            })
            env_overfit = EnvExtended(**env_overfit_params)
            print(f"Using Extended Environment for overfitting check with lead_time={args.lead_time}, history_length={args.history_length}")
        else:
            env_overfit = Env(**env_overfit_params)
            print("Using Original Environment for overfitting check")

        # Evaluate the *same trained agent* on this new environment
        print(f"\nEvaluating {agent_type.upper()} agent (trained on original env) on MODIFIED environment...")
        overfitting_eval_metrics = evaluate_agent(
            agent_type=f"{agent_type}_overfit_check", # Modified type for output file names
            env=env_overfit, # The new environment
            agent=agent,     # The original trained agent
            agent_index=target_firm_index,
            other_strategies=other_strategies_list_for_env, # Same other strategies
            num_episodes=100,
            max_steps=max_steps
        )

        print(f"Overfitting analysis for {agent_type.upper()} complete. Check results ending with '_overfit_check'.")
        print(f"--- End Overfitting Analysis for {agent_type.upper()} ---")
        # >>> END OVERFITTING ANALYSIS BLOCK <<<
    
    # Compare strategies including all trained agents
    print("\nComparing strategies...")
    comparison_data = compare_strategies( # Store the returned dict
        env=env,
        target_firm_index=target_firm_index,
        state_size=state_size, # MODIFIED: Pass state_size
        agent_types=args.agents,
        other_firm_strategy=args.other_strategy, # compare_strategies has its own logic for this based on name
        num_episodes=100, # Standard evaluation episodes
        max_steps=max_steps
    )

    # Later, this comparison_data will be passed to the new plotting function
    # For now, we can just print a confirmation or a small part of it.
    if comparison_data:
        print("\nStrategy comparison finished. Data collected for all strategies.")
        for strategy_name, metrics in comparison_data.items():
            avg_reward_val = metrics.get('avg_reward', np.nan)
            print(f"Summary for {strategy_name}: Avg Reward = {avg_reward_val:.2f}")

        print("\nGenerating comparison plots...")
        plot_performance_comparison(
            comparison_results=comparison_data,
            target_firm_index=args.firm,
            results_dir=os.path.join("results")
        )

if __name__ == "__main__":
    main()