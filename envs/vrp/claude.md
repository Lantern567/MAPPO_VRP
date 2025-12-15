# VRP 环境详细文档

## 概述

VRP (Vehicle Routing Problem) 环境实现了**卡车-无人机协同配送**场景。这是一个**多智能体合作**环境，卡车作为"母舰"沿路线行驶，无人机从卡车起飞完成配送任务。

---

## 文件说明

| 文件 | 功能 |
|-----|------|
| `core.py` | 定义所有实体类：Truck, Drone, Customer, World |
| `environment.py` | 环境类 `MultiAgentVRPEnv`，实现 Gym 接口 |
| `VRP_env.py` | 工厂函数，根据配置创建环境实例 |
| `env_wrappers.py` | 向量化环境包装器，支持并行采样 |
| `scenarios/` | 场景定义，包含奖励、观测、终止条件等 |

---

## 核心类详解

### World 类 (`core.py`)

```python
class World:
    """世界容器，包含所有实体"""

    def __init__(self):
        self.truck = None           # 卡车实例
        self.drones = []            # 无人机列表
        self.customers = []         # 客户列表
        self.route_nodes = []       # 卡车路线节点

        self.dim_p = 2              # 2D 位置空间
        self.dt = 0.1               # 物理仿真时间步长
        self.world_length = 200     # 最大 episode 步数
        self.world_step = 0         # 当前步数

        # 阈值
        self.delivery_threshold = 0.05   # 配送完成距离
        self.recovery_threshold = 0.1    # 无人机回收距离

    @property
    def policy_agents(self):
        """返回所有可控 Agent，顺序固定: [truck, drone_0, drone_1, ...]"""
        return [self.truck] + self.drones
```

### Truck 类 (`core.py`)

```python
class TruckState:
    """卡车状态"""
    p_pos = None           # 位置 [x, y]
    p_vel = None           # 速度 [vx, vy]
    current_node = None    # 当前所在路线节点索引

class TruckAction:
    """卡车动作"""
    move_target = None     # 移动目标位置
    release_drone = None   # 要释放的无人机索引
    recover_drone = None   # 要回收的无人机索引

class Truck:
    """卡车实体"""
    state = TruckState()
    action = TruckAction()
    max_speed = 1.0        # 最大速度
    drone_capacity = 3     # 最大载机数

    # 每步追踪
    distance_traveled_this_step = 0.0
```

### Drone 类 (`core.py`)

```python
class DroneState:
    """无人机状态"""
    p_pos = None                # 位置 [x, y]
    p_vel = None                # 速度 [vx, vy]
    battery = None              # 电量 (0.0 ~ 1.0)
    carrying_package = None     # 携带的包裹 (客户索引 或 None)
    status = None               # 状态: 'onboard'/'flying'/'returning'/'crashed'
    target_pos = None           # 当前目标位置

class DroneAction:
    """无人机动作"""
    target_customer = None      # 配送目标客户索引
    return_to_truck = False     # 是否返回卡车
    hover = False               # 是否悬停

class Drone:
    """无人机实体"""
    state = DroneState()
    action = DroneAction()
    max_speed = 2.0                    # 最大速度 (比卡车快)
    max_battery = 1.0                  # 满电
    battery_consumption_rate = 0.01   # 每单位距离耗电

    # 每步追踪
    battery_used_this_step = 0.0
    forced_return_this_step = False   # 是否被强制返航
```

### Customer 类 (`core.py`)

```python
class CustomerState:
    """客户状态"""
    p_pos = None               # 位置 [x, y]
    served = False             # 是否已服务
    demand = None              # 需求量 (0.0 ~ 1.0)
    time_window_start = None   # 时间窗开始 (step)
    time_window_end = None     # 时间窗结束 (step)
    arrival_step = None        # 实际送达时间

class Customer:
    """客户实体"""
    state = CustomerState()
    just_served_this_step = False  # 本步是否刚被服务 (用于奖励计算)
```

---

## 动作空间详解

### Truck 动作空间

```python
# Discrete(1 + num_nodes + 2 * num_drones)
# 例如: 5 个路线节点, 2 架无人机 → Discrete(10)

动作索引 | 含义
---------|------
0        | STAY (停留)
1~5      | MOVE_TO_NODE_0 ~ MOVE_TO_NODE_4 (移动到节点)
6~7      | RELEASE_DRONE_0 ~ RELEASE_DRONE_1 (释放无人机)
8~9      | RECOVER_DRONE_0 ~ RECOVER_DRONE_1 (回收无人机)
```

### Drone 动作空间

```python
# Discrete(2 + num_customers)
# 例如: 3 个客户 → Discrete(5)

动作索引 | 含义
---------|------
0        | HOVER (悬停)
1        | RETURN_TO_TRUCK (返回卡车)
2~4      | DELIVER_TO_CUSTOMER_0 ~ 2 (配送到客户)
```

---

## 观测空间详解

### 个体观测 (obs)

每个 Agent 看到的是**相对坐标**和局部信息：

**Drone 观测** (维度约 33，取决于配置):
```
[自身状态]
├── p_pos (2)           # 自身位置
├── p_vel (2)           # 自身速度
├── battery (1)         # 电量
├── carrying (1)        # 是否携带包裹
├── target_pos (2)      # 目标位置
├── onboard (1)         # 是否在卡车上
└── truck_rel_pos (2)   # 卡车相对位置

[客户信息] × num_customers
├── rel_pos (2)         # 客户相对位置
├── served (1)          # 是否已服务
├── tw_remain (1)       # 时间窗剩余
└── demand (1)          # 需求量

[其他无人机] × (num_drones - 1)
├── rel_pos (2)         # 相对位置
├── battery (1)         # 电量
└── status (1)          # 状态编码

[Agent ID]
└── one_hot (num_agents) # Agent 标识
```

**Truck 观测** (维度约 38，取决于配置):
```
[自身状态]
├── p_pos (2)           # 自身位置
└── p_vel (2)           # 自身速度

[无人机在位掩码]
└── onboard_mask (num_drones)

[无人机状态] × num_drones
├── rel_pos (2)         # 相对位置
├── p_vel (2)           # 速度
├── battery (1)         # 电量
├── carrying (1)        # 是否携带
└── status (1)          # 状态编码

[客户信息] × num_customers
├── rel_pos (2)
├── served (1)
├── tw_remain (1)
└── demand (1)

[Agent ID]
└── one_hot (num_agents)
```

### 共享观测 (share_obs)

所有 Agent 共享的**全局状态**，使用**绝对坐标**：

```
维度: 4 + 7×num_drones + 5×num_customers + 1

[Truck 状态] (4)
├── p_pos (2)           # 绝对位置
└── p_vel (2)           # 速度

[所有 Drone 状态] × num_drones (7 each)
├── p_pos (2)           # 绝对位置
├── p_vel (2)           # 速度
├── battery (1)
├── carrying (1)
└── status (1)

[所有 Customer 状态] × num_customers (5 each)
├── p_pos (2)           # 绝对位置
├── served (1)
├── tw_remain (1)
└── demand (1)

[时间] (1)
└── normalized_step     # current_step / world_length
```

**示例维度计算** (2 drones, 3 customers):
```
4 + 7×2 + 5×3 + 1 = 4 + 14 + 15 + 1 = 34
```

---

## Step 函数执行顺序

```python
def step(self, action_n):
    """环境单步执行 - 顺序严格固定！"""

    # 0. 重置本步临时标记
    self._reset_step_flags()

    # 1. 解析动作
    for agent, action in zip(self.agents, action_n):
        self._set_action(action, agent)

    # 2. 电量约束检查 (可能强制返航)
    self._enforce_battery_constraints()

    # 3. 处理卡车的释放/回收动作
    self._process_truck_release_recover()

    # 4. 更新无人机状态 (移动、耗电)
    self._update_drone_states()

    # 5. 更新卡车状态 (移动)
    self._update_truck_state()

    # 6. 检查配送完成
    self._check_deliveries()

    # 7. 推进世界时间
    self.world.step()

    # 8. 计算全局奖励 (只算一次！)
    global_reward = self._compute_global_reward()

    # 9. 收集返回值
    obs_n, reward_n, done_n, info_n = [], [], [], []
    ...

    # 10. 清理本步标记
    self._clear_step_flags()

    return obs_n, reward_n, done_n, info_n
```

---

## 关键机制

### 电量约束与强制返航

```python
def _enforce_battery_constraints(self):
    for drone in self.world.drones:
        if drone.state.status == 'onboard':
            continue

        # 计算返回卡车所需电量 (带 20% 安全余量)
        dist_to_truck = np.linalg.norm(drone.pos - truck.pos)
        battery_needed = dist_to_truck * consumption_rate * 1.2

        if drone.state.battery < battery_needed:
            # 强制返航！
            drone.action.return_to_truck = True
            drone.action.target_customer = None
            drone.forced_return_this_step = True  # 记录，用于惩罚
```

### 配送检查 (防止重复服务)

```python
def _check_deliveries(self):
    for drone in self.world.drones:
        if drone.action.target_customer is None:
            continue

        customer = self.world.customers[drone.action.target_customer]

        # 关键：跳过已服务的客户！
        if customer.state.served:
            continue

        dist = np.linalg.norm(drone.pos - customer.pos)
        if dist < self.world.delivery_threshold:
            customer.state.served = True
            customer.state.arrival_step = self.current_step
            customer.just_served_this_step = True
            drone.state.carrying_package = None
```

---

## 数据流图

```
┌─────────────┐     reset()      ┌─────────────┐
│   Scenario  │ ──────────────→  │    World    │
│ (参数配置)   │                  │  (实体状态)  │
└─────────────┘                  └──────┬──────┘
                                        │
                                        ↓
                              ┌─────────────────┐
                              │ MultiAgentVRPEnv │
                              │   (环境逻辑)     │
                              └────────┬────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ↓                          ↓                          ↓
     ┌──────────┐               ┌──────────┐               ┌──────────┐
     │  obs_n   │               │ reward_n │               │ info_n   │
     │ (观测)   │               │  (奖励)   │               │ (信息)   │
     └──────────┘               └──────────┘               └──────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ↓
                              ┌─────────────────┐
                              │   VRPRunner     │
                              │  (训练循环)      │
                              └─────────────────┘
```

---

## 使用示例

```python
from argparse import Namespace
from mappo.envs.vrp.VRP_env import VRPEnv

# 创建配置
args = Namespace(
    scenario_name='truck_drone_basic',
    num_drones=2,
    num_customers=3,
    num_route_nodes=5,
    episode_length=200,
    delivery_threshold=0.05,
    recovery_threshold=0.1
)

# 创建环境
env = VRPEnv(args)

# 查看空间信息
print(f"Agent 数量: {env.n}")
print(f"动作空间: {env.action_space}")
print(f"观测空间: {[s.shape for s in env.observation_space]}")
print(f"共享观测空间: {env.share_observation_space[0].shape}")

# 运行一个 episode
obs_n = env.reset()
for step in range(200):
    # 随机动作
    actions = [space.sample() for space in env.action_space]
    obs_n, reward_n, done_n, info_n = env.step(actions)

    if all(done_n):
        print(f"Episode 在第 {step} 步结束")
        break

env.close()
```

---

## 更多文档

- [场景设计](scenarios/claude.md) - 奖励函数、观测生成、动作掩码的详细说明
