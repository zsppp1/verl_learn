# verl 源码架构与走读指南

> 本文档梳理 verl 项目的源码架构，并提供推荐的代码阅读顺序。

## 一、目录结构总览

```
verl/
├── trainer/              # 训练入口层
│   ├── main_ppo.py       # PPO训练主入口
│   ├── ppo/              # PPO核心实现
│   │   ├── ray_trainer.py    # RayPPOTrainer（核心训练逻辑）
│   │   ├── core_algos.py     # PPO算法实现（损失函数、优势估计）
│   │   └── reward.py         # 奖励计算
│   └── config/           # 配置文件（YAML）
│
├── workers/              # 分布式Worker层
│   ├── fsdp_workers.py       # FSDP后端Worker（Actor/Critic/Reward）
│   ├── megatron_workers.py   # Megatron后端Worker
│   ├── engine_workers.py     # 新版引擎Worker
│   ├── rollout/              # Rollout生成
│   │   ├── vllm_rollout/     # vLLM集成
│   │   └── sglang_rollout/   # SGLang集成
│   └── sharding_manager/     # 分片管理
│
├── single_controller/    # 分布式控制层
│   ├── base/             # Worker基类和装饰器
│   └── ray/              # Ray实现
│       └── base.py       # RayWorkerGroup
│
├── models/               # 模型层
│   ├── transformers/     # HF模型适配
│   ├── mcore/            # Megatron-Core桥接
│   └── llama/qwen2/      # 特定模型实现
│
├── protocol.py           # DataProto（核心数据结构）
│
├── utils/                # 工具层
│   ├── dataset/          # 数据集处理
│   ├── checkpoint/       # 检查点管理
│   └── fsdp_utils/       # FSDP工具
│
└── experimental/         # 实验性功能
    ├── agent_loop/       # Agent循环
    └── reward_loop/      # 奖励循环
```

## 二、核心架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        main_ppo.py                               │
│  (入口：Hydra配置 → Ray初始化 → TaskRunner.run)                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TaskRunner                                   │
│  - 创建 Role-Worker Mapping                                      │
│  - 初始化 ResourcePoolManager (GPU资源池)                        │
│  - 创建 Dataset / RewardManager                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RayPPOTrainer                                  │
│  (verl/trainer/ppo/ray_trainer.py)                               │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ ActorRollout│  │   Critic    │  │  RewardModel│              │
│  │   Worker    │  │   Worker    │  │   Worker    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│        │                │                │                       │
│        ▼                ▼                ▼                       │
│  ┌─────────────────────────────────────────────────┐            │
│  │              RayWorkerGroup                      │            │
│  │    (管理Ray远程Actor，分布式调度)                │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  Training Loop:                                                  │
│  1. generate_sequences() → Rollout                              │
│  2. compute_reward() → Reward Model / Function                   │
│  3. compute_advantage() → GAE/GRPO                              │
│  4. update_policy() → PPO loss                                  │
│  5. update_critic() → Value loss                                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    core_algos.py                                 │
│  - compute_advantage() (GAE, GRPO, REINFORCE++)                  │
│  - ppo_loss() / grpo_loss()                                      │
│  - kl_penalty()                                                   │
│  - AdaptiveKLController                                          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DataProto                                    │
│  (verl/protocol.py)                                              │
│  - TensorDict封装，用于Worker间数据传输                           │
│  - batch, meta, non_tensor_batch                                │
└─────────────────────────────────────────────────────────────────┘
```

## 三、推荐走读顺序

### 第一阶段：理解入口和数据流

| 序号 | 文件路径 | 重点内容 |
|------|----------|----------|
| 1 | `verl/trainer/main_ppo.py` | 训练入口，理解整体流程、TaskRunner初始化 |
| 2 | `verl/protocol.py` | DataProto，核心数据结构，Worker间数据传输 |
| 3 | `verl/trainer/ppo/ray_trainer.py` | RayPPOTrainer，核心训练循环逻辑 |

### 第二阶段：理解PPO算法核心

| 序号 | 文件路径 | 重点内容 |
|------|----------|----------|
| 4 | `verl/trainer/ppo/core_algos.py` | PPO损失函数、优势估计（GAE/GRPO） |
| 5 | `verl/trainer/ppo/reward.py` | 奖励计算逻辑、RewardManager |

### 第三阶段：理解分布式Worker架构

| 序号 | 文件路径 | 重点内容 |
|------|----------|----------|
| 6 | `verl/single_controller/base/decorator.py` | `@register`装饰器机制，Worker方法注册 |
| 7 | `verl/single_controller/ray/base.py` | RayWorkerGroup，Ray分布式调度 |
| 8 | `verl/workers/fsdp_workers.py` | FSDP Worker实现（Actor/Critic/RewardModel） |

### 第四阶段：理解Rollout生成

| 序号 | 文件路径 | 重点内容 |
|------|----------|----------|
| 9 | `verl/workers/rollout/base.py` | Rollout基类接口定义 |
| 10 | `verl/workers/rollout/vllm_rollout/` | vLLM集成，高效推理生成 |
| 11 | `verl/workers/rollout/sglang_rollout/` | SGLang集成，结构化生成 |

### 第五阶段：理解配置和模型

| 序号 | 文件路径 | 重点内容 |
|------|----------|----------|
| 12 | `verl/trainer/config/ppo_trainer.yaml` | 配置文件结构，参数定义 |
| 13 | `verl/workers/config/` | Worker配置类（FSDPEngineConfig, RolloutConfig） |
| 14 | `verl/models/transformers/` | HuggingFace模型适配 |

## 四、核心类和函数速查表

| 模块 | 核心类/函数 | 作用 |
|------|------------|------|
| `main_ppo.py` | `TaskRunner.run()` | 初始化Worker、启动训练 |
| `ray_trainer.py` | `RayPPOTrainer` | 训练循环控制 |
| `ray_trainer.py` | `fit()` | 主训练循环实现 |
| `core_algos.py` | `compute_gae_advantage()` | GAE优势估计 |
| `core_algos.py` | `ppo_policy_loss()` | PPO策略损失函数 |
| `core_algos.py` | `AdaptiveKLController` | 自适应KL惩罚控制 |
| `fsdp_workers.py` | `ActorRolloutRefWorker` | Actor生成 + Reference Policy |
| `fsdp_workers.py` | `CriticWorker` | Critic价值网络 |
| `fsdp_workers.py` | `RewardModelWorker` | Reward Model推理 |
| `protocol.py` | `DataProto` | 数据传输协议（TensorDict封装） |

## 五、PPO训练流程详解

```
┌────────────────────────────────────────────────────────────┐
│                    PPO Training Loop                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  for epoch in range(max_epochs):                          │
│      │                                                     │
│      │  1. ──► generate_rollout()                         │
│      │       Actor模型生成response                         │
│      │       (使用vLLM/SGLang/HF)                          │
│      │                                                     │
│      │  2. ──► compute_reward()                           │
│      │       Reward Model / Rule-based reward             │
│      │       可选：KL penalty                              │
│      │                                                     │
│      │  3. ──► compute_advantage()                        │
│      │       GAE: A = Σ(γλ)^t * δ_t                        │
│      │       或 GRPO: group normalization                  │
│      │                                                     │
│      │  4. ──► update_policy()                            │
│      │       L = min(r(θ)*A, clip(r(θ))*A)                │
│      │       多轮更新 (ppo_epochs)                         │
│      │                                                     │
│      │  5. ──► update_critic()                            │
│      │       L = MSE(V, returns)                          │
│      │                                                     │
│      │  6. ──► save_checkpoint()                          │
│      │                                                     │
│      └──► next epoch                                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 各步骤详解

#### Step 1: generate_rollout()
- **位置**: `RayPPOTrainer.generate_sequences()`
- **作用**: 使用Actor模型生成response
- **后端**: vLLM / SGLang / HuggingFace Transformers
- **数据流**: `prompt → Actor → sequences + log_probs`

#### Step 2: compute_reward()
- **位置**: `RayPPOTrainer.compute_reward()`
- **作用**: 计算生成序列的奖励值
- **类型**:
  - Model-based: RewardModelWorker
  - Rule-based: 自定义reward函数
  - KL penalty: 参考策略与当前策略KL散度

#### Step 3: compute_advantage()
- **位置**: `core_algos.py`
- **方法**:
  - GAE (Generalized Advantage Estimation)
  - GRPO (Group Relative Policy Optimization)
  - REINFORCE++

#### Step 4: update_policy()
- **位置**: `RayPPOTrainer.update_policy()`
- **算法**: PPO clipped objective
- **公式**: `L = min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)`
- **迭代**: ppo_epochs次

#### Step 5: update_critic()
- **位置**: `RayPPOTrainer.update_critic()`
- **目标**: 最小化价值估计误差
- **公式**: `L = MSE(V(s), returns)`

## 六、关键设计模式

### 1. HybridFlow编程模型

verl 采用混合控制器编程模型：

- **单控制器模式** (`single_controller`): 集中调度，类似传统的参数服务器
- **多控制器模式** (`RayWorkerGroup`): 分布式执行，每个Worker独立运行

```python
# 单控制器：集中调度
trainer = RayPPOTrainer(...)
trainer.fit()  # 控制整个训练流程

# 多控制器：分布式Worker
actor_worker = ActorRolloutRefWorker.remote()
ray.get(actor_worker.generate_sequences.remote(data))
```

### 2. Worker注册机制

使用装饰器注册Worker方法，支持分布式调用：

```python
# verl/single_controller/base/decorator.py
from verl.single_controller.base.decorator import register, Dispatch

class ActorRolloutRefWorker(Worker):
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, data: DataProto):
        """生成rollout序列"""
        ...
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_policy(self, data: DataProto):
        """更新策略参数"""
        ...
```

### 3. 数据流转协议

使用 `DataProto` 作为统一数据传输协议：

```python
# verl/protocol.py
class DataProto:
    batch: TensorDict       # 主要数据（tokens, log_probs等）
    meta: dict              # 元数据信息
    non_tensor_batch: dict  # 非Tensor数据

# 数据流转示例
data = DataProto(batch=prompts)
data = actor_worker.generate_sequences(data)  # 添加response
data = reward_worker.compute_reward(data)     # 添加reward
data = trainer.compute_advantage(data)        # 添加advantage
```

### 4. 资源池管理

GPU资源通过 ResourcePoolManager 统一管理：

```python
# verl/trainer/ppo/ray_trainer.py
resource_pool_spec = {
    "global_pool": [8] * 4,  # 4节点，每节点8GPU
    "reward_pool": [4] * 2,  # 独立reward资源池
}

resource_pool_manager = ResourcePoolManager(resource_pool_spec)
```

## 七、配置系统

### 配置文件结构

```yaml
# verl/trainer/config/ppo_trainer.yaml

data:
  train_files: ["data/train.jsonl"]
  val_files: ["data/val.jsonl"]
  max_prompt_length: 512
  
actor_rollout_ref:
  model:
    path: "path/to/model"
  actor:
    strategy: "fsdp"  # fsdp / megatron
  rollout:
    backend: "vllm"   # vllm / sglang / hf

critic:
  strategy: "fsdp"
  
algorithm:
  adv_estimator: "gae"
  ppo_epochs: 4
  clip_ratio: 0.2
  use_kl_in_reward: false

trainer:
  total_epochs: 10
  nnodes: 1
  n_gpus_per_node: 8
```

### 配置类定义

```python
# verl/workers/config/
class FSDPEngineConfig:
    strategy: str = "fsdp"
    fsdp_size: int = -1
    
class RolloutConfig:
    backend: str = "vllm"
    temperature: float = 1.0
    top_p: float = 0.9
    
class AlgoConfig:
    adv_estimator: str = "gae"
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
```

## 八、扩展指南

### 添加新的优势估计方法

```python
# verl/trainer/ppo/core_algos.py

from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("my_advantage")
def compute_my_advantage(critic_values, rewards, gamma):
    """自定义优势估计"""
    ...
```

### 添加新的Rollout后端

```python
# verl/workers/rollout/my_rollout/

class MyRollout(RolloutBase):
    def generate_sequences(self, prompts):
        """自定义生成逻辑"""
        ...
```

### 添加新的模型支持

```python
# verl/models/transformers/my_model.py

class MyModelConfig:
    """模型配置"""
    
def load_my_model(config):
    """模型加载逻辑"""
```

## 九、常见问题

### Q1: 如何选择 FSDP vs Megatron？

- **FSDP**: 适合中小模型（<10B），配置简单
- **Megatron**: 适合大模型（>10B），需要更多配置

### Q2: 如何选择 vLLM vs SGLang？

- **vLLM**: 成熟稳定，吞吐量大
- **SGLang**: 结构化生成能力强，RadixAttention优化

### Q3: 如何调试分布式训练？

1. 使用 `ray.timeline()` 生成时间线
2. 检查 `ResourcePoolManager` 资源分配
3. 使用 `DataProto.meta` 跟踪数据流转

## 十、参考资源

- [verl GitHub](https://github.com/verl-project/verl)
- [HybridFlow Paper (EuroSys 2025)](https://arxiv.org/abs/...)
- [PPO Paper (OpenAI 2017)](https://arxiv.org/abs/1707.06347)
- [DeepWiki verl 文档](https://deepwiki.com/verl-project/verl)

---

> 本文档基于 verl v0.7.0 版本梳理