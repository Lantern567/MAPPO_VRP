# MAPPO VRP 项目文档

## 项目简介

本项目使用 **MAPPO (Multi-Agent Proximal Policy Optimization)** 算法解决 **VRP (Vehicle Routing Problem)** 车辆路径问题，具体场景为**卡车-无人机协同配送**。

### 问题背景

在现代物流配送中，卡车可以携带多架无人机进行"最后一公里"配送：
- **卡车 (Truck)**: 作为"母舰"，沿预定路线行驶，负责释放/回收无人机
- **无人机 (Drone)**: 从卡车起飞，快速配送包裹到客户位置，完成后返回卡车
- **客户 (Customer)**: 有固定位置、包裹需求和时间窗约束

### 目标

通过多智能体强化学习，让卡车和无人机学会协同配送，最小化：
- 总配送时间
- 能源消耗
- 时间窗违约惩罚

---

## 项目架构

```
mappo/
├── __init__.py              # 包初始化
├── claude.md                # 本文档 - 项目总览
│
├── envs/                    # 环境定义
│   ├── __init__.py
│   ├── claude.md            # 环境包文档
│   └── vrp/                 # VRP 环境实现
│       ├── __init__.py
│       ├── claude.md        # VRP 环境详细文档
│       ├── core.py          # 实体类 (Truck, Drone, Customer, World)
│       ├── environment.py   # 环境类 MultiAgentVRPEnv
│       ├── VRP_env.py       # 环境工厂函数
│       ├── env_wrappers.py  # 向量化环境包装器
│       └── scenarios/       # 场景定义
│           ├── __init__.py
│           ├── claude.md    # 场景文档
│           └── truck_drone_basic.py  # 基础场景
│
├── runner/                  # 训练运行器
│   ├── __init__.py
│   ├── claude.md            # Runner 文档
│   └── vrp_runner.py        # VRP 专用 Runner
│
├── scripts/                 # 训练脚本
│   ├── __init__.py
│   ├── claude.md            # 脚本使用文档
│   └── train_vrp.py         # 训练入口脚本
│
└── results/                 # 训练结果 (自动生成)
    └── VRP/
        └── {scenario}/{algorithm}/{experiment}/run{N}/
            ├── logs/        # TensorBoard 日志
            └── models/      # 保存的模型权重
```

---

## 核心概念

### 1. Multi-Agent (多智能体)

本项目是一个**合作式多智能体**系统：
- **Agent 数量**: 1 辆卡车 + N 架无人机 (默认 N=2)
- **Agent 顺序**: 固定为 `[truck, drone_0, drone_1, ...]`
- **合作关系**: 所有 Agent 共享同一个全局奖励

### 2. CTDE (中心化训练，去中心化执行)

```
训练时 (Centralized):
┌─────────────────────────────────────────┐
│         Centralized Critic              │
│    (使用全局状态 share_obs 估计 V)        │
└─────────────────────────────────────────┘
        ↑                    ↑
   ┌────┴────┐          ┌────┴────┐
   │ Truck   │          │ Drone   │
   │ Actor   │          │ Actor   │
   └────┬────┘          └────┬────┘
        ↓                    ↓
   局部观测 obs         局部观测 obs

执行时 (Decentralized):
每个 Agent 只根据自己的局部观测 obs 做决策
```

### 3. Separated Policy (分离策略)

由于卡车和无人机的**动作空间不同**，必须使用分离策略模式：

| Agent | 动作空间维度 | 动作类型 |
|-------|------------|---------|
| Truck | 1 + num_nodes + 2×num_drones | 停留、移动、释放、回收 |
| Drone | 2 + num_customers | 悬停、返回、配送到客户X |

- **Policy 0**: 卡车策略 (1个)
- **Policy 1**: 无人机共享策略 (所有无人机共用)

---

## 快速开始

### 环境要求

```bash
# 依赖包
- Python 3.8+
- PyTorch 1.8+
- gymnasium
- numpy
- tensorboardX
- setproctitle
```

### 训练命令

```bash
# 进入项目根目录
cd rl_logistics

# 基础训练 (使用默认参数)
python mappo/scripts/train_vrp.py \
    --scenario_name truck_drone_basic \
    --algorithm_name mappo

# 完整训练 (自定义参数)
python mappo/scripts/train_vrp.py \
    --scenario_name truck_drone_basic \
    --algorithm_name mappo \
    --num_drones 2 \
    --num_customers 3 \
    --episode_length 200 \
    --n_rollout_threads 8 \
    --num_env_steps 1000000 \
    --hidden_size 128 \
    --use_centralized_V True \
    --experiment_name my_experiment
```

### 查看训练日志

```bash
tensorboard --logdir mappo/results/VRP/truck_drone_basic/mappo/
```

---

## 依赖关系

### 外部依赖

本项目依赖 [on-policy](../on-policy/) MAPPO 框架：

```
mappo/                      on-policy/
  │                            │
  ├── runner/                  ├── onpolicy/
  │   └── vrp_runner.py ────→  │   └── runner/separated/base_runner.py
  │                            │
  └── scripts/                 ├── onpolicy/config.py
      └── train_vrp.py ────→   └── onpolicy/algorithms/r_mappo/
```

### 关键导入

```python
# 从 on-policy 框架导入
from onpolicy.runner.separated.base_runner import Runner
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
```

---

## 关键设计决策

### 1. Agent 顺序固定

```python
self.agents = [world.truck] + world.drones  # 永远是这个顺序！
# agents[0] = truck
# agents[1] = drone_0
# agents[2] = drone_1
```

### 2. 全局奖励 (只算一次)

```python
# 正确 ✓
global_reward = compute_global_reward(world)
reward_n = [[global_reward]] * num_agents

# 错误 ✗ (会导致重复累加)
reward_n = [compute_reward(agent, world) for agent in agents]
```

### 3. 观测空间 Padding

由于卡车和无人机观测维度不同，为了兼容向量化环境，所有观测都 padding 到最大维度：

```python
max_obs_dim = max(truck_obs_dim, drone_obs_dim)
# truck_obs: [actual_obs, 0, 0, 0, ...]  padding to max_obs_dim
# drone_obs: [actual_obs, 0, 0, ...]      padding to max_obs_dim
```

### 4. 动作掩码 (Action Mask)

禁止无效动作，例如：
- 已服务的客户不能再配送
- 在卡车上的无人机只能等待被释放
- 只能回收距离足够近的无人机

---

## 常见问题

### Q1: 为什么使用 separated policy 而不是 shared policy?

因为卡车和无人机的动作空间维度不同：
- Truck: `Discrete(10)` (1个停留 + 5个移动 + 2×2个释放/回收)
- Drone: `Discrete(5)` (1个悬停 + 1个返回 + 3个配送)

如果使用 shared policy，所有 agent 必须有相同的动作空间。

### Q2: share_obs 是什么?

`share_obs` 是**全局状态**，用于中心化 Critic 估计价值函数：
- 包含所有实体的**绝对坐标**
- 所有 Agent 的 share_obs **完全相同**
- 维度: `4 (truck) + 7×num_drones + 5×num_customers + 1 (time)`

### Q3: 训练时 FPS 太低怎么办?

1. 增加并行环境数: `--n_rollout_threads 16`
2. 使用 GPU: `--cuda True`
3. 减少 episode_length 进行快速验证: `--episode_length 50`

---

## 更多文档

- [环境包文档](envs/claude.md)
- [VRP 环境详解](envs/vrp/claude.md)
- [场景设计文档](envs/vrp/scenarios/claude.md)
- [Runner 文档](runner/claude.md)
- [训练脚本文档](scripts/claude.md)
