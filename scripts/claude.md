# 训练脚本文档

## 概述

`scripts/` 目录包含启动 MAPPO 训练的入口脚本。

---

## 文件说明

| 文件 | 功能 |
|-----|------|
| `__init__.py` | 包初始化 |
| `train_vrp.py` | VRP 环境训练入口脚本 |

---

## train_vrp.py 详解

### 基本用法

```bash
# 最简命令
python mappo/scripts/train_vrp.py

# 指定场景和算法
python mappo/scripts/train_vrp.py \
    --scenario_name truck_drone_basic \
    --algorithm_name mappo
```

### 完整训练命令

```bash
python mappo/scripts/train_vrp.py \
    --scenario_name truck_drone_basic \
    --algorithm_name mappo \
    --num_drones 2 \
    --num_customers 3 \
    --num_route_nodes 5 \
    --episode_length 200 \
    --n_rollout_threads 8 \
    --num_env_steps 1000000 \
    --hidden_size 128 \
    --use_centralized_V \
    --experiment_name my_experiment
```

---

## 参数详解

### VRP 专用参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--scenario_name` | str | `truck_drone_basic` | 场景名称 |
| `--num_drones` | int | 2 | 无人机数量 |
| `--num_customers` | int | 3 | 客户数量 |
| `--num_route_nodes` | int | 5 | 卡车路线节点数 |
| `--delivery_threshold` | float | 0.05 | 配送完成距离阈值 |
| `--recovery_threshold` | float | 0.1 | 无人机回收距离阈值 |

### 奖励参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--delivery_bonus` | float | 10.0 | 成功配送奖励 |
| `--late_penalty` | float | 0.5 | 每步迟到惩罚 |
| `--energy_cost` | float | 0.1 | 能源消耗成本 |
| `--completion_bonus` | float | 50.0 | 全部完成额外奖励 |

### 算法参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--algorithm_name` | str | `mappo` | 算法名称: mappo, rmappo, ippo |
| `--use_centralized_V` | bool | True | 使用中心化 Critic |
| `--use_recurrent_policy` | bool | False | 使用 RNN 策略 |
| `--hidden_size` | int | 64 | 网络隐藏层大小 |
| `--lr` | float | 5e-4 | 学习率 |
| `--gamma` | float | 0.99 | 折扣因子 |
| `--gae_lambda` | float | 0.95 | GAE lambda |
| `--ppo_epoch` | int | 15 | PPO 更新轮数 |
| `--clip_param` | float | 0.2 | PPO clip 参数 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--episode_length` | int | 200 | 每轮最大步数 |
| `--num_env_steps` | int | 1000000 | 总训练步数 |
| `--n_rollout_threads` | int | 8 | 并行环境数 |
| `--n_training_threads` | int | 1 | 训练线程数 |
| `--use_linear_lr_decay` | bool | False | 使用学习率衰减 |

### 评估参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--use_eval` | bool | False | 是否启用评估 |
| `--eval_interval` | int | 25 | 评估间隔 (episode 数) |
| `--n_eval_rollout_threads` | int | 1 | 评估环境数 |

### 日志和保存参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--experiment_name` | str | `check` | 实验名称 |
| `--log_interval` | int | 5 | 日志打印间隔 |
| `--save_interval` | int | 1 | 模型保存间隔 |
| `--use_wandb` | bool | False | 使用 Weights & Biases |
| `--user_name` | str | `marl` | 用户名 (wandb) |

### 硬件参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--cuda` | bool | True | 使用 GPU |
| `--cuda_deterministic` | bool | True | CUDA 确定性 |
| `--seed` | int | 1 | 随机种子 |

---

## 算法选择

### MAPPO (推荐)

```bash
python mappo/scripts/train_vrp.py \
    --algorithm_name mappo \
    --use_centralized_V
```

- 使用中心化 Critic (share_obs)
- 去中心化 Actor (个体 obs)
- 不使用 RNN

### RMAPPO (带 RNN)

```bash
python mappo/scripts/train_vrp.py \
    --algorithm_name rmappo
```

- 自动启用 RNN 策略
- 适合部分可观测场景

### IPPO

```bash
python mappo/scripts/train_vrp.py \
    --algorithm_name ippo
```

- 去中心化 Critic (个体 obs)
- 完全独立训练
- 不推荐用于合作场景

---

## 脚本流程

```python
def main(args):
    # 1. 解析参数
    parser = get_config()
    all_args = parse_args(args, parser)

    # 2. 强制 VRP 使用 separated policy
    all_args.share_policy = False

    # 3. 配置算法
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        all_args.use_centralized_V = False

    # 4. 设置设备
    device = torch.device("cuda:0") if cuda else torch.device("cpu")

    # 5. 创建输出目录
    run_dir = results/VRP/{scenario}/{algorithm}/{experiment}/run{N}/

    # 6. 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 7. 创建环境
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args)

    # 8. 创建 Runner 并训练
    runner = VRPRunner(config)
    runner.run()

    # 9. 清理
    envs.close()
```

---

## 输出目录结构

```
mappo/results/
└── VRP/
    └── truck_drone_basic/          # 场景名
        └── mappo/                   # 算法名
            └── my_experiment/       # 实验名
                └── run1/            # 自动编号
                    ├── logs/        # TensorBoard 日志
                    │   └── events.out.tfevents.*
                    └── models/      # 保存的模型
                        ├── actor_agent0.pt
                        ├── actor_agent1.pt
                        ├── critic_agent0.pt
                        └── critic_agent1.pt
```

---

## 查看训练日志

### TensorBoard

```bash
tensorboard --logdir mappo/results/VRP/truck_drone_basic/mappo/
```

可视化指标:
- `average_episode_rewards`: 平均回报
- `policy_loss`: 策略损失
- `value_loss`: 价值损失
- `entropy`: 策略熵

### 控制台输出

```
VRP Scenario truck_drone_basic Algo mappo Exp my_experiment
Episode 10/500, Steps 16000/1000000, FPS 1234

  Customers served: 2.50/3.00 (83.3%)
  Agent 0 average episode rewards: 45.67
  Agent 1 average episode rewards: 45.67
  Agent 2 average episode rewards: 45.67
```

---

## Weights & Biases 集成

```bash
# 启用 wandb
python mappo/scripts/train_vrp.py \
    --use_wandb \
    --user_name your_username

# 首次使用需要登录
wandb login
```

---

## 环境创建函数

### `make_train_env(all_args)`

```python
def make_train_env(all_args):
    """创建向量化训练环境"""
    def get_env_fn(rank):
        def init_env():
            env = VRPEnv(all_args)
            env.seed(all_args.seed + rank * 1000)  # 每个环境不同种子
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return VRPDummyVecEnv([get_env_fn(0)])  # 单环境
    else:
        return VRPSubprocVecEnv([get_env_fn(i) for i in range(n)])  # 多进程
```

### `make_eval_env(all_args)`

```python
def make_eval_env(all_args):
    """创建评估环境"""
    # 使用不同的种子区间，避免与训练环境重叠
    env.seed(all_args.seed * 50000 + rank * 10000)
```

---

## 常见用例

### 快速测试

```bash
python mappo/scripts/train_vrp.py \
    --num_env_steps 10000 \
    --episode_length 50 \
    --n_rollout_threads 2 \
    --log_interval 1
```

### 增加难度

```bash
python mappo/scripts/train_vrp.py \
    --num_drones 3 \
    --num_customers 5 \
    --episode_length 300
```

### 使用 GPU + 大规模训练

```bash
python mappo/scripts/train_vrp.py \
    --cuda \
    --n_rollout_threads 32 \
    --num_env_steps 10000000 \
    --hidden_size 256
```

### 带评估的训练

```bash
python mappo/scripts/train_vrp.py \
    --use_eval \
    --eval_interval 10 \
    --n_eval_rollout_threads 4
```

---

## 常见问题

### Q1: 训练速度太慢?

A: 尝试以下方法:
1. 增加并行环境: `--n_rollout_threads 16`
2. 使用 GPU: `--cuda`
3. 减少网络大小: `--hidden_size 64`

### Q2: 奖励不收敛?

A: 检查:
1. 学习率是否合适: 尝试 `--lr 1e-4` 或 `--lr 1e-3`
2. 增加训练步数: `--num_env_steps 5000000`
3. 调整奖励参数: `--delivery_bonus 20 --late_penalty 1.0`

### Q3: 如何恢复训练?

A: 目前不直接支持断点续训。需要:
1. 手动加载保存的模型权重
2. 或使用 wandb 的 resume 功能

### Q4: 内存不足?

A: 减少并行环境数和网络大小:
```bash
--n_rollout_threads 4 --hidden_size 64
```

### Q5: 多 GPU 训练?

A: 目前不支持。可以在不同 GPU 上运行多个实验:
```bash
CUDA_VISIBLE_DEVICES=0 python train_vrp.py --seed 1 &
CUDA_VISIBLE_DEVICES=1 python train_vrp.py --seed 2 &
```
