# 环境包 (envs) 文档

## 概述

`envs/` 目录包含所有强化学习环境的定义。环境遵循 **Gymnasium** (原 OpenAI Gym) 接口规范，提供标准的 `reset()` 和 `step()` 方法。

---

## 目录结构

```
envs/
├── __init__.py          # 包初始化，导出 VRPEnv
├── claude.md            # 本文档
└── vrp/                 # VRP (车辆路径问题) 环境
    ├── __init__.py
    ├── core.py          # 实体类定义
    ├── environment.py   # 环境类
    ├── VRP_env.py       # 工厂函数
    ├── env_wrappers.py  # 向量化包装器
    └── scenarios/       # 场景定义
```

---

## 当前支持的环境

| 环境名称 | 描述 | Agent 数量 |
|---------|------|-----------|
| VRP | 卡车-无人机协同配送 | 1 + num_drones |

---

## 环境接口规范

所有环境都实现以下接口：

```python
class MultiAgentEnv:
    """多智能体环境基类"""

    def __init__(self, world, ...):
        self.n = num_agents                    # Agent 数量
        self.agents = [...]                    # Agent 列表
        self.action_space = [...]              # 每个 Agent 的动作空间
        self.observation_space = [...]         # 每个 Agent 的观测空间
        self.share_observation_space = [...]   # 共享观测空间

    def reset(self) -> List[np.ndarray]:
        """重置环境，返回初始观测"""
        return obs_n  # shape: (n_agents, obs_dim)

    def step(self, action_n) -> Tuple:
        """执行动作，返回下一状态"""
        return obs_n, reward_n, done_n, info_n

    def seed(self, seed: int):
        """设置随机种子"""
        pass

    def close(self):
        """关闭环境，释放资源"""
        pass
```

---

## 返回值格式

### obs_n (观测)
```python
# 形状: List[np.ndarray]
# 长度: n_agents
# 每个元素: (obs_dim,) 的一维数组
obs_n = [
    np.array([...]),  # agent 0 (truck) 的观测
    np.array([...]),  # agent 1 (drone_0) 的观测
    np.array([...]),  # agent 2 (drone_1) 的观测
]
```

### reward_n (奖励)
```python
# 形状: List[List[float]]
# 合作场景下所有 Agent 奖励相同
reward_n = [
    [global_reward],  # agent 0
    [global_reward],  # agent 1
    [global_reward],  # agent 2
]
```

### done_n (终止标志)
```python
# 形状: List[bool]
# 合作场景下所有 Agent 同时终止
done_n = [True, True, True]  # 或 [False, False, False]
```

### info_n (附加信息)
```python
# 形状: List[dict]
info_n = [
    {
        'available_actions': np.array([...]),  # 动作掩码
        'share_obs': np.array([...]),          # 全局状态
        'policy_id': 0,                        # 策略 ID
        'customers_served': 2,                 # VRP 特有
        'total_customers': 3,                  # VRP 特有
    },
    {...},
    {...},
]
```

---

## 向量化环境

为了并行采样，环境被包装成向量化版本：

```python
# 单环境 (用于调试)
from mappo.envs.vrp.env_wrappers import VRPDummyVecEnv
envs = VRPDummyVecEnv([lambda: VRPEnv(args)])

# 多进程并行 (用于训练)
from mappo.envs.vrp.env_wrappers import VRPSubprocVecEnv
envs = VRPSubprocVecEnv([lambda: VRPEnv(args) for _ in range(8)])
```

向量化后的返回值增加一个环境维度：
```python
obs = envs.reset()      # shape: (n_envs, n_agents, obs_dim)
obs, rewards, dones, infos = envs.step(actions)
# obs:     (n_envs, n_agents, obs_dim)
# rewards: (n_envs, n_agents, 1)
# dones:   (n_envs, n_agents)
# infos:   List[List[dict]]
```

---

## 如何添加新环境

1. 在 `envs/` 下创建新目录，例如 `envs/new_env/`

2. 实现必要文件：
   ```
   envs/new_env/
   ├── __init__.py
   ├── core.py          # 实体定义
   ├── environment.py   # 环境类
   └── scenarios/       # 场景
   ```

3. 确保环境类实现标准接口

4. 在 `envs/__init__.py` 中导出新环境

5. 创建对应的 Runner (如需要)

---

## 更多文档

- [VRP 环境详解](vrp/claude.md) - 卡车-无人机配送环境的完整文档
- [场景设计](vrp/scenarios/claude.md) - 如何定义配送场景
