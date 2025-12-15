# Runner 文档

## 概述

`runner/` 目录包含 MAPPO 训练的运行器 (Runner)，负责管理整个训练循环：数据采集、策略更新、日志记录等。

---

## 文件说明

| 文件 | 功能 |
|-----|------|
| `__init__.py` | 包初始化 |
| `vrp_runner.py` | VRP 专用 Runner，继承自 on-policy 框架 |

---

## VRPRunner 类详解

### 继承关系

```
on-policy/onpolicy/runner/separated/base_runner.py
                    │
                    └── Runner (基类)
                            │
                            └── VRPRunner (本项目)
```

### 为什么使用 Separated Runner?

VRP 环境中，卡车和无人机的**动作空间维度不同**：
- Truck: `Discrete(1 + num_nodes + 2*num_drones)` ≈ 10 维
- Drone: `Discrete(2 + num_customers)` ≈ 5 维

因此必须使用 **Separated Policy Mode**，为每种 Agent 类型维护独立的策略网络。

---

## 核心方法

### `__init__(self, config)`

初始化 Runner，接收配置字典：

```python
config = {
    "all_args": all_args,      # 所有参数
    "envs": envs,               # 训练环境 (向量化)
    "eval_envs": eval_envs,     # 评估环境
    "num_agents": num_agents,   # Agent 数量
    "device": device,           # CPU/GPU
    "run_dir": run_dir          # 输出目录
}
```

### `run(self)` - 主训练循环

```python
def run(self):
    """主训练循环"""
    self.warmup()  # 初始化 buffer

    episodes = num_env_steps // episode_length // n_rollout_threads

    for episode in range(episodes):
        # 1. 学习率衰减
        if self.use_linear_lr_decay:
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].policy.lr_decay(episode, episodes)

        # 2. 采集数据 (Rollout)
        for step in range(self.episode_length):
            values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
            obs, rewards, dones, infos = self.envs.step(actions_env)
            self.insert(data)

        # 3. 计算 returns (GAE)
        self.compute()

        # 4. 训练 (PPO 更新)
        train_infos = self.train()

        # 5. 保存模型 & 日志
        if episode % self.save_interval == 0:
            self.save()
        if episode % self.log_interval == 0:
            self._log_vrp_metrics(infos, train_infos)
```

### `warmup(self)` - 初始化 Buffer

```python
def warmup(self):
    """
    初始化 replay buffer 的第一个观测

    流程:
    1. reset 环境获取初始 obs
    2. 做一次 dummy step 获取 share_obs (在 info 中)
    3. 再次 reset 获取干净的初始状态
    4. 将 obs 和 share_obs 存入 buffer[0]
    """
    obs = self.envs.reset()

    # Dummy step 获取 share_obs
    dummy_actions = [[one_hot(0)] * num_agents] * n_rollout_threads
    _, _, _, infos = self.envs.step(dummy_actions)
    share_obs = self._get_share_obs_from_infos(infos)

    # 重新 reset
    obs = self.envs.reset()

    # 初始化 buffer
    for agent_id in range(self.num_agents):
        self.buffer[agent_id].share_obs[0] = share_obs.copy()
        self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
```

### `collect(self, step)` - 采集动作

```python
@torch.no_grad()
def collect(self, step):
    """
    从策略网络采样动作

    返回:
        values: 价值估计 [envs, agents, 1]
        actions: 动作索引 [envs, agents, 1]
        action_log_probs: 动作对数概率 [envs, agents, 1]
        rnn_states: Actor RNN 状态
        rnn_states_critic: Critic RNN 状态
        actions_env: one-hot 编码的动作，用于环境 step
    """
    for agent_id in range(self.num_agents):
        # 从 buffer 获取当前观测
        value, action, action_log_prob, rnn_state, rnn_state_critic = \
            self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],  # Critic 输入
                self.buffer[agent_id].obs[step],         # Actor 输入
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step]
            )

        # 转换为 one-hot 供环境使用
        action_env = np.eye(action_dim)[action]
```

### `insert(self, data)` - 插入数据到 Buffer

```python
def insert(self, data):
    """
    将采集到的数据插入 replay buffer

    处理:
    1. 在 done 时重置 RNN 状态
    2. 创建 masks (done=0, not_done=1)
    3. 从 info 中提取 share_obs
    4. 为每个 agent 插入数据
    """
    obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

    # Reset RNN states on done
    rnn_states[dones == True] = np.zeros(...)
    rnn_states_critic[dones == True] = np.zeros(...)

    # Create masks
    masks = np.ones((n_rollout_threads, num_agents, 1))
    masks[dones == True] = 0

    # Get share_obs from info
    share_obs = self._get_share_obs_from_infos(infos)

    # Insert for each agent
    for agent_id in range(self.num_agents):
        self.buffer[agent_id].insert(
            share_obs,
            obs[:, agent_id],
            rnn_states[:, agent_id],
            rnn_states_critic[:, agent_id],
            actions[:, agent_id],
            action_log_probs[:, agent_id],
            values[:, agent_id],
            rewards[:, agent_id],
            masks[:, agent_id]
        )
```

### `_get_share_obs_from_infos(self, infos)` - 提取共享观测

```python
def _get_share_obs_from_infos(self, infos):
    """
    从 info 字典中提取 share_obs

    VRP 中所有 agent 共享相同的全局状态:
    - Truck 绝对位置/速度
    - 所有 Drone 绝对位置/速度/电量/状态
    - 所有 Customer 位置/服务状态/时间窗
    - 当前时间步
    """
    share_obs_list = []
    for env_infos in infos:
        # 所有 agent 的 share_obs 相同，取第一个即可
        if 'share_obs' in env_infos[0]:
            share_obs_list.append(env_infos[0]['share_obs'])
        else:
            # Fallback
            share_obs_list.append(np.zeros(share_obs_dim))
    return np.array(share_obs_list)
```

### `_log_vrp_metrics(self, infos, train_infos)` - 日志记录

```python
def _log_vrp_metrics(self, infos, train_infos):
    """
    记录 VRP 特定指标:
    - customers_served: 已服务客户数
    - total_customers: 总客户数
    - completion_rate: 完成率
    - average_episode_rewards: 每个 agent 的平均回报
    """
    # 从 info 中提取指标
    customers_served = []
    total_customers = []

    for env_infos in infos:
        for agent_info in env_infos:
            if 'customers_served' in agent_info:
                customers_served.append(agent_info['customers_served'])
            if 'total_customers' in agent_info:
                total_customers.append(agent_info['total_customers'])

    # 计算并打印
    completion_rate = np.mean(customers_served) / np.mean(total_customers)
    print(f"  Customers served: {np.mean(customers_served):.2f}/{np.mean(total_customers):.2f} "
          f"({completion_rate * 100:.1f}%)")
```

### `eval(self, total_num_steps)` - 评估

```python
@torch.no_grad()
def eval(self, total_num_steps):
    """
    评估当前策略

    特点:
    - 使用 deterministic=True 采样确定性动作
    - 使用独立的 eval_envs
    - 记录 eval_average_episode_rewards
    """
```

---

## 训练流程图

```
                    ┌─────────────┐
                    │   warmup()  │
                    │ 初始化 buffer │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   for episode in ...   │◄──────────┐
              └───────────┬────────────┘           │
                          │                        │
                          ▼                        │
          ┌───────────────────────────────┐        │
          │     for step in episode:      │        │
          │  ┌─────────────────────────┐  │        │
          │  │    collect(step)       │  │        │
          │  │  从策略采样动作          │  │        │
          │  └───────────┬─────────────┘  │        │
          │              │                │        │
          │              ▼                │        │
          │  ┌─────────────────────────┐  │        │
          │  │    envs.step(actions)   │  │        │
          │  │    环境交互             │  │        │
          │  └───────────┬─────────────┘  │        │
          │              │                │        │
          │              ▼                │        │
          │  ┌─────────────────────────┐  │        │
          │  │    insert(data)         │  │        │
          │  │    存储到 buffer        │  │        │
          │  └─────────────────────────┘  │        │
          └───────────────┬───────────────┘        │
                          │                        │
                          ▼                        │
              ┌───────────────────────┐            │
              │      compute()        │            │
              │    计算 GAE returns   │            │
              └───────────┬───────────┘            │
                          │                        │
                          ▼                        │
              ┌───────────────────────┐            │
              │       train()         │            │
              │     PPO 策略更新      │            │
              └───────────┬───────────┘            │
                          │                        │
                          ▼                        │
              ┌───────────────────────┐            │
              │   save() / log()      │            │
              │   保存模型 / 日志     │────────────┘
              └───────────────────────┘
```

---

## 数据格式

### Buffer 结构

每个 Agent 有独立的 Buffer:

```python
self.buffer[agent_id] = SeparatedReplayBuffer(
    args,
    obs_space,           # 个体观测空间
    share_obs_space,     # 共享观测空间
    act_space            # 动作空间
)
```

Buffer 存储的数据:

| 字段 | 形状 | 说明 |
|-----|------|------|
| share_obs | [T+1, N, share_obs_dim] | 全局状态 |
| obs | [T+1, N, obs_dim] | 个体观测 |
| rnn_states | [T+1, N, recurrent_N, hidden_size] | Actor RNN |
| rnn_states_critic | [T+1, N, recurrent_N, hidden_size] | Critic RNN |
| actions | [T, N, 1] | 动作索引 |
| action_log_probs | [T, N, 1] | 动作概率 |
| value_preds | [T+1, N, 1] | 价值预测 |
| rewards | [T, N, 1] | 奖励 |
| masks | [T+1, N, 1] | Done 掩码 |
| returns | [T+1, N, 1] | 计算的 returns |

其中: T = episode_length, N = n_rollout_threads

---

## 关键设计

### 1. Separated Policy

```python
# 每个 agent 有独立的 trainer
self.trainer = []
for agent_id in range(num_agents):
    tr = TrainAlgo(...)  # R_MAPPO 或 R_IPPO
    self.trainer.append(tr)

# Policy 0: Truck
# Policy 1: Drone (所有无人机共享)
# agent_id 0 -> trainer[0] (truck policy)
# agent_id 1,2,... -> trainer[1,2,...] (实际训练时可配置共享)
```

### 2. Centralized Critic

```python
# 所有 agent 的 Critic 看到相同的 share_obs
for agent_id in range(self.num_agents):
    if self.use_centralized_V:
        share_obs_agent = share_obs  # 全局状态
    else:
        share_obs_agent = obs[:, agent_id]  # 个体观测
```

### 3. 动作格式转换

```python
# 策略输出: action index [envs, 1]
# 环境需要: one-hot [envs, action_dim]

action_env = np.eye(action_dim)[action]

# 然后重排为 [envs, agents, action_dim]
actions_env = []
for i in range(n_rollout_threads):
    actions_env.append([temp_actions_env[a][i] for a in range(num_agents)])
```

---

## 与 on-policy 框架的关系

```
on-policy/onpolicy/
├── runner/
│   └── separated/
│       └── base_runner.py    ← VRPRunner 继承此类
│
├── algorithms/
│   └── r_mappo/
│       └── r_mappo.py        ← 训练算法
│
├── utils/
│   └── separated_buffer.py   ← Replay Buffer
│
└── config.py                  ← 参数配置
```

VRPRunner 覆盖的方法:
- `run()`: 添加 VRP 特定日志
- `warmup()`: 处理 share_obs 初始化
- `collect()`: 动作格式转换
- `insert()`: 从 info 提取 share_obs
- `eval()`: VRP 评估逻辑

---

## 常见问题

### Q1: 为什么 share_obs 在 info 中而不是直接返回?

A: Gymnasium 接口的 `step()` 只返回 `(obs, reward, done, info)`。share_obs 是额外信息，放在 info 中是标准做法。这样也兼容不需要 share_obs 的算法。

### Q2: 如何添加自定义日志指标?

A: 在 `_log_vrp_metrics()` 方法中添加：

```python
def _log_vrp_metrics(self, infos, train_infos):
    # 现有指标...

    # 添加新指标
    if 'my_metric' in infos[0][0]:
        my_metric = np.mean([i[0]['my_metric'] for i in infos])
        print(f"  My Metric: {my_metric:.2f}")

        # 写入 TensorBoard
        self.writter.add_scalar('vrp/my_metric', my_metric, total_num_steps)
```

### Q3: 如何调整训练频率?

A: 修改相关参数：
- `--save_interval`: 模型保存间隔 (episode 数)
- `--log_interval`: 日志打印间隔
- `--eval_interval`: 评估间隔
