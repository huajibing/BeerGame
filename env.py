import numpy as np
import collections

class Env:
    def __init__(self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100):
        """
        初始化供应链管理仿真环境。
        
        :param num_firms: 企业数量
        :param p: 各企业的价格列表
        :param h: 库存持有成本
        :param c: 损失销售成本
        :param initial_inventory: 每个企业的初始库存
        :param poisson_lambda: 最下游企业需求的泊松分布均值
        :param max_steps: 每个episode的最大步数
        """
        self.num_firms = num_firms
        self.p = p  # 企业的价格列表
        self.h = h  # 库存持有成本
        self.c = c  # 损失销售成本
        self.poisson_lambda = poisson_lambda  # 泊松分布的均值
        self.max_steps = max_steps  # 每个episode的最大步数
        self.initial_inventory = initial_inventory  # 初始库存
        
        # 初始化库存
        self.inventory = np.full((num_firms, 1), initial_inventory)
        # 初始化订单量
        self.orders = np.zeros((num_firms, 1))
        # 初始化已满足的需求量
        self.satisfied_demand = np.zeros((num_firms, 1))
        # 记录当前步数
        self.current_step = 0
        # 标记episode是否结束
        self.done = False
        self.state_size = 3 # For compatibility with dynamic state sizing in beer_game.py

    def reset(self):
        """
        重置环境状态。
        """
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory)
        self.orders = np.zeros((self.num_firms, 1))
        self.satisfied_demand = np.zeros((self.num_firms, 1))
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """
        获取每个企业的观察信息，包括订单量、满足的需求量和库存。
        每个企业的状态是独立的，包括自己观察的订单、需求和库存。
        """
        return np.concatenate((self.orders, self.satisfied_demand, self.inventory), axis=1)

    def _generate_demand(self):
        """
        根据规则生成每个企业的需求。
        最下游企业的需求遵循泊松分布，其他企业的需求等于下游企业的订单量。
        """
        demand = np.zeros((self.num_firms, 1))
        for i in range(self.num_firms):
            if i == 0:
                # 最下游企业的需求遵循泊松分布，均值为 poisson_lambda
                demand[i] = np.random.poisson(self.poisson_lambda)
            else:
                # 上游企业的需求等于下游企业的订单量
                demand[i] = self.orders[i - 1]  # d_{i+1,t} = q_{it}
        return demand

    def step(self, actions):
        """
        执行一个时间步的仿真，根据给定的行动 (每个企业的订单量) 更新环境状态。
        
        :param actions: 每个企业的订单量 (shape: (num_firms, 1))，即每个智能体的行动
        :return: next_state, reward, done
        """
        self.orders = actions  # 更新订单量
        
        # 生成各企业的需求
        self.demand = self._generate_demand()

        # 计算每个企业收到的订单量和满足的需求
        for i in range(self.num_firms):
            if i == 0:
                # 第一企业从外部需求直接得到满足
                self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
            else:
                # 后续企业的需求由上游企业订单决定
                self.satisfied_demand[i] = min(self.demand[i], self.inventory[i])
        
        # 更新库存
        for i in range(self.num_firms):
            self.inventory[i] = self.inventory[i] + self.orders[i] - self.satisfied_demand[i]
        
        # 计算每个企业的奖励: p_i * d_{it} - p_{i+1} * q_{it} - h * I_{it}
        rewards = np.zeros((self.num_firms, 1))
        loss_sales = np.zeros((self.num_firms, 1))  # 损失销售费用
        
        for i in range(self.num_firms):
            rewards[i] += self.p[i] * self.satisfied_demand[i] - (self.p[i+1] if i+1 < self.num_firms else 0) * self.orders[i] - self.h * self.inventory[i]
            
            # 损失销售计算
            if self.satisfied_demand[i] < self.demand[i]:
                loss_sales[i] = (self.demand[i] - self.satisfied_demand[i]) * self.c
        
        rewards -= loss_sales  # 总奖励扣除损失销售成本
        
        # 增加步数
        self.current_step += 1
        
        # 判断是否结束（比如达到最大步数）
        if self.current_step >= self.max_steps:
            self.done = True
        
        return self._get_observation(), rewards, self.done

# EnvExtended class definition moved above this line
class EnvExtended(Env):
    def __init__(self, num_firms, p, h, c, initial_inventory, poisson_lambda=10, max_steps=100, lead_time=2, history_length=3):
        super().__init__(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)
        self.lead_time = lead_time
        self.history_length = history_length
        self.in_transit_pipelines = [collections.deque() for _ in range(self.num_firms)]
        self.historical_orders = [collections.deque([0.0] * history_length, maxlen=history_length) for _ in range(self.num_firms)]
        self.historical_inventory = [collections.deque([float(self.initial_inventory)] * history_length, maxlen=history_length) for _ in range(self.num_firms)]
        self.historical_demand = [collections.deque([0.0] * history_length, maxlen=history_length) for _ in range(self.num_firms)]
        self.historical_backlog = [collections.deque([0.0] * history_length, maxlen=history_length) for _ in range(self.num_firms)]
        self.current_backlog = np.zeros((self.num_firms, 1), dtype=np.float32)
        self.state_size = 7  # Fixed based on the planned extended state

    def reset(self):
        # Base class variables explicitly reset here
        self.inventory = np.full((self.num_firms, 1), self.initial_inventory, dtype=np.float32)
        self.orders = np.zeros((self.num_firms, 1), dtype=np.float32)
        self.satisfied_demand = np.zeros((self.num_firms, 1), dtype=np.float32)
        self.current_step = 0
        self.done = False
        self.demand = np.zeros((self.num_firms, 1), dtype=np.float32) # Initialize self.demand

        # Initialize EnvExtended specific states
        self.in_transit_pipelines = [collections.deque() for _ in range(self.num_firms)]
        self.current_backlog = np.zeros((self.num_firms, 1), dtype=np.float32)

        # Initialize historical deques
        for i in range(self.num_firms):
            self.historical_inventory[i] = collections.deque([float(self.initial_inventory)] * self.history_length, maxlen=self.history_length)
            self.historical_orders[i] = collections.deque([0.0] * self.history_length, maxlen=self.history_length)
            self.historical_demand[i] = collections.deque([0.0] * self.history_length, maxlen=self.history_length)
            self.historical_backlog[i] = collections.deque([0.0] * self.history_length, maxlen=self.history_length)

        return self._get_observation()

    def _update_history(self):
        for i in range(self.num_firms):
            self.historical_orders[i].append(self.orders[i][0])
            self.historical_inventory[i].append(self.inventory[i][0])
            self.historical_demand[i].append(self.demand[i][0]) # demand is set in step method
            self.historical_backlog[i].append(self.current_backlog[i][0])

    def step(self, actions):
        self.current_step += 1

        # Shipment Arrival
        for firm_idx in range(self.num_firms):
            arrived_today = 0.0
            remaining_pipeline = collections.deque()
            while self.in_transit_pipelines[firm_idx]: # Use while to process all items
                quantity, arrival_step = self.in_transit_pipelines[firm_idx].popleft() # More efficient
                if arrival_step == self.current_step:
                    arrived_today += quantity
                elif arrival_step > self.current_step:
                    remaining_pipeline.append((quantity, arrival_step))
            self.in_transit_pipelines[firm_idx] = remaining_pipeline # Assign back only those not yet arrived
            self.inventory[firm_idx] += arrived_today

        self.orders = np.array(actions, dtype=np.float32).reshape(self.num_firms, 1)
        for firm_idx in range(self.num_firms):
            if self.orders[firm_idx][0] > 0:  # Only add positive orders to pipeline
                self.in_transit_pipelines[firm_idx].append((self.orders[firm_idx][0], self.current_step + self.lead_time))
                # Sort pipeline by arrival step - important if lead_time can be variable or if items are added out of order
                self.in_transit_pipelines[firm_idx] = collections.deque(sorted(list(self.in_transit_pipelines[firm_idx]), key=lambda x: x[1]))

        self.demand = self._generate_demand() # Generate demand after orders are placed for upstream firms

        # Satisfy Demand & Update Inventory
        self.satisfied_demand = np.zeros((self.num_firms, 1), dtype=np.float32) # Reset satisfied demand for the step
        for firm_idx in range(self.num_firms):
            available_inventory = self.inventory[firm_idx][0]
            demand_firm = self.demand[firm_idx][0]

            satisfied_this_step = min(demand_firm, available_inventory)
            self.satisfied_demand[firm_idx] = satisfied_this_step
            self.inventory[firm_idx] -= satisfied_this_step

        self.current_backlog = self.demand - self.satisfied_demand
        self.current_backlog[self.current_backlog < 0] = 0 # Ensure backlog is not negative

        # Calculate Rewards
        rewards = np.zeros((self.num_firms, 1), dtype=np.float32)
        loss_sales_cost = np.zeros((self.num_firms, 1), dtype=np.float32)
        for i in range(self.num_firms):
            rewards[i] += self.p[i] * self.satisfied_demand[i]
            # Cost of placing an order with the upstream supplier
            rewards[i] -= (self.p[i+1] if i+1 < self.num_firms else 0) * self.orders[i]
            rewards[i] -= self.h * self.inventory[i]
            loss_sales_cost[i] = self.current_backlog[i] * self.c
        rewards -= loss_sales_cost

        self._update_history()

        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_observation(), rewards, self.done

    def _get_observation(self):
        # This method now acts as a router based on some condition or can be directly overridden
        return self._get_observation_extended()

    def _get_observation_extended(self):
        obs_all_firms = []
        for i in range(self.num_firms):
            s_inventory = self.inventory[i][0]
            s_backlog = self.current_backlog[i][0]
            s_in_transit_total = sum(item[0] for item in self.in_transit_pipelines[i])
            s_next_arrival_quantity = sum(item[0] for item in self.in_transit_pipelines[i] if item[1] == self.current_step + 1)

            # Demand for firm i is self.demand[i][0]. This is what firm i *sees* as its incoming orders.
            s_downstream_orders = self.demand[i][0]

            # Downstream backlog for firm i is the backlog of firm i-1.
            s_downstream_backlog = self.historical_backlog[i-1][-1] if i > 0 else 0.0

            # Upstream inventory for firm i is the inventory of firm i+1.
            s_upstream_inventory = self.historical_inventory[i+1][-1] if i < self.num_firms - 1 else 0.0

            obs_firm_i = [
                s_inventory,
                s_backlog,
                s_in_transit_total,
                s_next_arrival_quantity,
                s_downstream_orders,
                s_downstream_backlog,
                s_upstream_inventory
            ]
            obs_all_firms.append(obs_firm_i)
        return np.array(obs_all_firms, dtype=np.float32)


# 使用示例
if __name__ == "__main__":
    # 初始化环境
    num_firms = 3  # 假设有3个企业
    p = [10, 9, 8]  # 价格列表
    h = 0.5  # 库存持有成本
    c = 2  # 损失销售成本
    initial_inventory = 100  # 初始库存
    poisson_lambda = 10  # 泊松分布的均值
    max_steps = 100  # 每个episode的最大步数

    # 创建仿真环境
    env = Env(num_firms, p, h, c, initial_inventory, poisson_lambda, max_steps)

    # 进行多个episode的仿真
    for episode in range(5):  # 假设进行5个episode
        state = env.reset()
        total_rewards = np.zeros((num_firms, 1))  # 每个企业的总奖励
        done = False
        # while not done: # Original Env test loop
            # 假设每个企业的订单量是随机的，这里可以换成更复杂的策略
            # actions = np.random.randint(1, 21, size=(num_firms, 1))  # 随机生成每个企业的订单量
            # next_state, rewards, done = env.step(actions)
            # total_rewards += rewards
            # print(f"Episode {episode + 1}, Step {env.current_step}, Rewards: {rewards.T}, Total Rewards: {total_rewards.T}")
        if episode == 0: # Only print details for the first episode of original env test
            print(f"Original Env - Episode {episode + 1}, Initial State:\n{state}")
            actions_sample = np.random.randint(1, 21, size=(num_firms, 1))
            next_state_sample, rewards_sample, done_sample = env.step(actions_sample)
            print(f"Original Env - Episode {episode + 1}, Step 1 Actions:\n{actions_sample}")
            print(f"Original Env - Episode {episode + 1}, Step 1 Next State:\n{next_state_sample}")
            print(f"Original Env - Episode {episode + 1}, Step 1 Rewards:\n{rewards_sample.T}")
            print(f"Original Env - Episode {episode + 1}, Step 1 Done: {done_sample}")
        break # Only run one episode for the original Env test to keep output concise


    print("\n\n--- Testing EnvExtended ---")
    num_firms_ext = 3
    p_ext = [10, 9, 8]
    h_ext = 0.5
    c_ext = 2
    initial_inventory_ext = 50
    poisson_lambda_ext = 5
    max_steps_ext = 10
    lead_time_ext = 2
    history_length_ext = 3

    env_ext = EnvExtended(
        num_firms_ext, p_ext, h_ext, c_ext, initial_inventory_ext,
        poisson_lambda_ext, max_steps_ext, lead_time_ext, history_length_ext
    )

    print(f"EnvExtended initialized with: lead_time={lead_time_ext}, history_length={history_length_ext}")

    initial_obs_ext = env_ext.reset()
    print(f"\nInitial Observation (EnvExtended):\n{initial_obs_ext}")
    # Expected shape: (num_firms, 7)
    # Columns: s_inventory, s_backlog, s_in_transit_total, s_next_arrival_quantity,
    #          s_downstream_orders, s_downstream_backlog, s_upstream_inventory

    num_test_steps = 4
    for step_num in range(1, num_test_steps + 1):
        random_actions = np.random.randint(1, 11, size=(num_firms_ext, 1)) # Smaller orders for testing
        print(f"\n--- Step {step_num} ---")
        print(f"Actions taken:\n{random_actions}")

        next_state, rewards, done = env_ext.step(random_actions)

        print(f"Next State (Observation):\n{next_state}")
        print(f"Rewards:\n{rewards.T}")
        print(f"Done: {done}")

        print("\nDetailed state components for each firm:")
        for i in range(num_firms_ext):
            print(f"  Firm {i}:")
            print(f"    Inventory: {next_state[i, 0]:.1f}")
            print(f"    Backlog: {next_state[i, 1]:.1f}")
            print(f"    In-Transit Total: {next_state[i, 2]:.1f}")
            print(f"    Next Arrival Quantity (at t+1): {next_state[i, 3]:.1f}")
            print(f"    Downstream Orders (Demand for this firm): {next_state[i, 4]:.1f}")
            print(f"    Downstream Backlog (Firm i-1's backlog): {next_state[i, 5]:.1f}")
            print(f"    Upstream Inventory (Firm i+1's inventory): {next_state[i, 6]:.1f}")
            print(f"    Current Inventory (direct from env): {env_ext.inventory[i][0]:.1f}")
            print(f"    In-Transit Pipeline: {list(env_ext.in_transit_pipelines[i])}")


        if step_num == num_test_steps -1 : # Print history towards the end
            print("\nHistorical Data for Firm 0 (after step " + str(step_num) + "):")
            print(f"  Historical Orders: {list(env_ext.historical_orders[0])}")
            print(f"  Historical Inventory: {[round(inv,1) for inv in env_ext.historical_inventory[0]]}")
            print(f"  Historical Demand: {list(env_ext.historical_demand[0])}")
            print(f"  Historical Backlog: {list(env_ext.historical_backlog[0])}")

        if done:
            print("Episode finished early.")
            break
