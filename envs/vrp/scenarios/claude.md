# VRP 场景设计文档

## 概述

`scenarios/` 目录定义了具体的配送场景。每个场景是一个 Python 模块，包含一个 `Scenario` 类，负责：
- 创建世界和实体
- 重置世界状态
- 生成观测
- 计算奖励
- 提供动作掩码
- 判断终止条件

---

## 文件说明

| 文件 | 功能 |
|-----|------|
| `__init__.py` | 场景加载器，动态导入场景模块 |
| `truck_drone_basic.py` | 基础卡车-无人机配送场景 (MVP) |

---

## Scenario 类接口

每个场景必须实现以下方法：

```python
class Scenario:
    """场景基类接口"""

    def __init__(self):
        """初始化奖励参数等"""
        self.delivery_bonus = 10.0
        self.late_penalty = 0.5
        ...

    def make_world(self, args) -> World:
        """创建并返回 World 实例"""
        pass

    def reset_world(self, world):
        """重置世界到初始状态"""
        pass

    def observation(self, agent, world) -> np.ndarray:
        """生成 agent 的局部观测"""
        pass

    def get_share_obs(self, world) -> np.ndarray:
        """生成全局共享观测"""
        pass

    def compute_global_reward(self, world) -> float:
        """计算全局奖励 (只调用一次！)"""
        pass

    def get_available_actions(self, agent, world) -> np.ndarray:
        """返回动作掩码 (1=可用, 0=禁用)"""
        pass

    def is_terminal(self, world) -> bool:
        """判断 episode 是否终止"""
        pass

    def info(self, agent, world) -> dict:
        """返回附加信息"""
        pass
```

---

## truck_drone_basic.py 详解

### 奖励函数

```python
class Scenario:
    def __init__(self):
        # 奖励参数
        self.delivery_bonus = 10.0       # 成功配送奖励
        self.late_penalty = 0.5          # 每步迟到惩罚
        self.energy_cost = 0.1           # 能源消耗成本
        self.completion_bonus = 50.0     # 全部完成额外奖励
        self.incomplete_penalty = 10.0   # 每个未完成客户惩罚
        self.forced_return_penalty = 1.0 # 强制返航惩罚

    def compute_global_reward(self, world):
        """
        计算全局奖励 - 所有 Agent 共享
        注意：只调用一次，避免重复累加！
        """
        rew = 0.0

        # 1. 配送奖励 + 时间窗惩罚
        for customer in world.customers:
            if customer.just_served_this_step:
                rew += self.delivery_bonus

                # 迟到惩罚
                late_steps = max(0,
                    customer.state.arrival_step - customer.state.time_window_end
                )
                rew -= self.late_penalty * late_steps

        # 2. 能源消耗
        for drone in world.drones:
            rew -= self.energy_cost * drone.battery_used_this_step

        # 3. 强制返航惩罚
        for drone in world.drones:
            if drone.forced_return_this_step:
                rew -= self.forced_return_penalty

        # 4. 终止奖励/惩罚
        if self.is_terminal(world):
            served = sum(1 for c in world.customers if c.state.served)
            total = len(world.customers)

            if served == total:
                rew += self.completion_bonus
            else:
                rew -= self.incomplete_penalty * (total - served)

        return rew
```

**奖励组成分析**:

| 奖励项 | 值 | 触发条件 |
|-------|-----|---------|
| 配送成功 | +10.0 | 无人机到达客户位置 |
| 迟到惩罚 | -0.5/step | 到达时间 > 时间窗结束 |
| 能源消耗 | -0.1×距离 | 无人机移动 |
| 强制返航 | -1.0 | 电量不足被迫返航 |
| 全部完成 | +50.0 | 所有客户都被服务 |
| 未完成惩罚 | -10.0×数量 | episode 结束时未服务的客户 |

---

### 观测生成

```python
def observation(self, agent, world):
    """生成 agent 的局部观测"""
    obs = []

    # 1. 自身状态 (所有 agent 共有)
    obs.extend(agent.state.p_pos)  # [x, y]
    obs.extend(agent.state.p_vel)  # [vx, vy]

    if isinstance(agent, Drone):
        # 2. Drone 特有信息
        obs.append(agent.state.battery)
        obs.append(1.0 if agent.state.carrying_package else 0.0)

        if agent.state.target_pos is not None:
            obs.extend(agent.state.target_pos)
        else:
            obs.extend([0.0, 0.0])

        obs.append(1.0 if agent.state.status == 'onboard' else 0.0)

        # 卡车相对位置
        rel_truck = world.truck.state.p_pos - agent.state.p_pos
        obs.extend(rel_truck)

        # 所有客户的相对信息
        for customer in world.customers:
            rel_pos = customer.state.p_pos - agent.state.p_pos
            obs.extend(rel_pos)
            obs.append(1.0 if customer.state.served else 0.0)
            obs.append(self._time_window_remaining(customer, world))
            obs.append(customer.state.demand)

        # 其他无人机
        for other in world.drones:
            if other is agent:
                continue
            rel_pos = other.state.p_pos - agent.state.p_pos
            obs.extend(rel_pos)
            obs.append(other.state.battery)
            obs.append(self._encode_drone_status(other.state.status))

    else:  # Truck
        # 无人机在位掩码
        for drone in world.drones:
            obs.append(1.0 if drone.state.status == 'onboard' else 0.0)

        # 所有无人机状态
        for drone in world.drones:
            rel_pos = drone.state.p_pos - agent.state.p_pos
            obs.extend(rel_pos)
            obs.extend(drone.state.p_vel)
            obs.append(drone.state.battery)
            obs.append(1.0 if drone.state.carrying_package else 0.0)
            obs.append(self._encode_drone_status(drone.state.status))

        # 所有客户状态
        for customer in world.customers:
            rel_pos = customer.state.p_pos - agent.state.p_pos
            obs.extend(rel_pos)
            obs.append(1.0 if customer.state.served else 0.0)
            obs.append(self._time_window_remaining(customer, world))
            obs.append(customer.state.demand)

    # Agent ID one-hot
    agent_idx = world.policy_agents.index(agent)
    agent_id = [0.0] * len(world.policy_agents)
    agent_id[agent_idx] = 1.0
    obs.extend(agent_id)

    return np.array(obs, dtype=np.float32)
```

---

### 动作掩码

```python
def get_available_actions(self, agent, world):
    """返回动作掩码 (1=可用, 0=禁用)"""

    if isinstance(agent, Drone):
        # [HOVER, RETURN, DELIVER_0, DELIVER_1, ...]
        mask = np.ones(2 + len(world.customers))

        if agent.state.status == 'onboard':
            # 在卡车上：只能 HOVER (等待被释放)
            mask[1:] = 0

        elif agent.state.status == 'crashed':
            # 坠毁：只能 HOVER (什么也做不了)
            mask[:] = 0
            mask[0] = 1

        else:
            # 飞行中
            # 禁用已服务的客户
            for i, c in enumerate(world.customers):
                if c.state.served:
                    mask[2 + i] = 0

            # 如果正在携带包裹，只能送给目标客户
            if agent.state.carrying_package is not None:
                for i in range(len(world.customers)):
                    if i != agent.state.carrying_package:
                        mask[2 + i] = 0

    else:  # Truck
        # [STAY, MOVE_0..N, RELEASE_0..D, RECOVER_0..D]
        num_nodes = len(world.route_nodes)
        num_drones = len(world.drones)
        mask = np.ones(1 + num_nodes + 2 * num_drones)

        # RELEASE: 只能释放 onboard 的无人机
        for i, drone in enumerate(world.drones):
            if drone.state.status != 'onboard':
                mask[1 + num_nodes + i] = 0

        # RECOVER: 只能回收近距离且不在卡车上的无人机
        for i, drone in enumerate(world.drones):
            dist = np.linalg.norm(drone.state.p_pos - agent.state.p_pos)
            if dist > world.recovery_threshold or drone.state.status == 'onboard':
                mask[1 + num_nodes + num_drones + i] = 0

    return mask
```

**掩码逻辑总结**:

| Agent | 条件 | 禁用的动作 |
|-------|-----|-----------|
| Drone (onboard) | 在卡车上 | RETURN, 所有 DELIVER |
| Drone (flying) | 客户已服务 | 对应的 DELIVER |
| Drone (flying) | 携带包裹 | 其他客户的 DELIVER |
| Drone (crashed) | 坠毁 | 除 HOVER 外所有 |
| Truck | 无人机不在卡车上 | 对应的 RELEASE |
| Truck | 无人机太远或已在卡车上 | 对应的 RECOVER |

---

### 终止条件

```python
def is_terminal(self, world):
    """判断 episode 是否终止"""

    # 1. 达到最大步数
    if world.world_step >= world.world_length:
        return True

    # 2. 所有客户都已服务
    if all(c.state.served for c in world.customers):
        return True

    # 3. 所有无人机都坠毁
    if all(d.state.status == 'crashed' for d in world.drones):
        return True

    return False
```

---

### 世界初始化

```python
def make_world(self, args):
    """创建世界"""
    world = World()
    world.world_length = args.episode_length  # 默认 200

    # 创建卡车
    world.truck = Truck()
    world.truck.name = 'truck_0'

    # 创建无人机
    for i in range(args.num_drones):
        drone = Drone()
        drone.name = f'drone_{i}'
        drone.state.status = 'onboard'
        world.drones.append(drone)

    # 创建客户
    for i in range(args.num_customers):
        customer = Customer()
        customer.name = f'customer_{i}'
        world.customers.append(customer)

    # 生成卡车路线节点 (圆形分布)
    world.route_nodes = self._generate_route_nodes(args.num_route_nodes)

    # 初始化
    self.reset_world(world)
    return world

def reset_world(self, world):
    """重置世界状态"""
    # 卡车回到原点
    world.truck.state.p_pos = np.array([0.0, 0.0])
    world.truck.state.p_vel = np.zeros(2)

    # 无人机回到卡车上
    for drone in world.drones:
        drone.state.p_pos = world.truck.state.p_pos.copy()
        drone.state.battery = drone.max_battery
        drone.state.status = 'onboard'
        drone.state.carrying_package = None

    # 随机生成客户位置和时间窗
    for customer in world.customers:
        customer.state.p_pos = np.random.uniform(-0.8, 0.8, 2)
        customer.state.served = False
        customer.state.demand = np.random.uniform(0.5, 1.0)

        # 时间窗 (单位: step)
        tw_start = np.random.randint(0, world.world_length // 2)
        tw_duration = np.random.randint(world.world_length // 3, world.world_length)
        customer.state.time_window_start = tw_start
        customer.state.time_window_end = min(tw_start + tw_duration, world.world_length)

    world.world_step = 0
```

---

## 如何创建新场景

1. 在 `scenarios/` 下创建新文件，例如 `my_scenario.py`

2. 实现 `Scenario` 类：
```python
class Scenario:
    def __init__(self):
        # 自定义奖励参数
        pass

    def make_world(self, args):
        # 自定义世界创建
        pass

    def reset_world(self, world):
        # 自定义重置逻辑
        pass

    def observation(self, agent, world):
        # 自定义观测
        pass

    # ... 其他必要方法
```

3. 训练时指定场景名称：
```bash
python train_vrp.py --scenario_name my_scenario
```

---

## 辅助方法

```python
def _time_window_remaining(self, customer, world):
    """计算归一化的时间窗剩余时间"""
    if customer.state.served:
        return 0.0
    remaining = (customer.state.time_window_end - world.world_step) / world.world_length
    return max(0.0, min(1.0, remaining))

def _encode_drone_status(self, status):
    """将无人机状态编码为浮点数"""
    status_map = {
        'onboard': 0.0,
        'flying': 0.25,
        'returning': 0.5,
        'crashed': 1.0
    }
    return status_map.get(status, 0.0)

def _generate_route_nodes(self, num_nodes):
    """生成卡车路线节点 (圆形分布)"""
    nodes = []
    for i in range(num_nodes):
        angle = 2 * np.pi * i / num_nodes
        x = 0.6 * np.cos(angle)
        y = 0.6 * np.sin(angle)
        nodes.append(np.array([x, y]))
    return nodes
```
