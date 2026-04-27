# add_critic_worker 函数解析

> verl/trainer/main_ppo.py 第167-192行

## 一、函数定义

```python
def add_critic_worker(self, config):
    """Add critic worker to role mapping."""
```

**核心职责：**
- 根据配置选择合适的 Critic Worker 类
- 注册 Role.Critic → Worker 的映射关系
- 将 Critic Worker 分配到 global_pool 资源池

---

## 二、决策流程图

```
                ┌─────────────────────────────────┐
                │   critic.strategy = ?           │
                └─────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
          "fsdp/fsdp2"    "megatron"      其他
              │              │              │
              ▼              ▼              ▼
    ┌─────────────────┐ ┌──────────┐ ┌──────────────┐
    │ use_legacy = ?  │ │ Megatron │ │ NotImplementedError
    │                 │ │ Critic   │ │              │
    │ ┌─────┬─────┐   │ │ Worker   │ │              │
    │ │auto │enable│  │ └──────────┘ │              │
    │ │     │     │   │              │              │
    │ │FSDP │FSDP │   │              │              │
    │ │Critic│Critic│ │              │              │
    │ │Work │Work │   │              │              │
    │ │er   │er   │   │              │              │
    │ └─────┴─────┘   │              │              │
    │                 │              │              │
    │ │disable│       │              │              │
    │ │       │       │              │              │
    │ │Training│      │              │              │
    │ │Worker │       │              │              │
    │ │(新版) │       │              │              │
    │ └───────┘       │              │              │
    └─────────────────┘              │              │
              │                      │              │
              ▼                      ▼              ▼
    ┌──────────────────────────────────────────────────┐
    │  注册到 role_worker_mapping                      │
    │  Role.Critic → ray.remote(Worker)                │
    │  mapping[Role.Critic] = "global_pool"            │
    └──────────────────────────────────────────────────┘
```

---

## 三、详细代码解析

### 3.1 配置读取

```python
use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
```

与 `add_actor_rollout_worker` 相同的配置项，决定使用新版引擎还是旧版 Worker。

---

### 3.2 FSDP/FSDP2 策略处理

```python
if config.critic.strategy in {"fsdp", "fsdp2"}:
    if use_legacy_worker_impl in ["auto", "enable"]:
        from verl.workers.fsdp_workers import CriticWorker
        
    elif use_legacy_worker_impl == "disable":
        # we don't need to specialize critic worker. Just use TrainingWorker
        from verl.workers.engine_workers import TrainingWorker
        CriticWorker = TrainingWorker
        print("Using new worker implementation")
```

**新旧版本对比：**

| 版本 | Worker类 | 文件位置 | 特点 |
|------|----------|----------|------|
| 旧版 (auto/enable) | CriticWorker | fsdp_workers.py | 专用的Critic训练Worker |
| 新版 (disable) | TrainingWorker | engine_workers.py | 通用训练Worker，更灵活 |

**为什么新版用 TrainingWorker？**

源码注释原文：
> "we don't need to specialize critic worker. Just use TrainingWorker"

新版引擎的设计理念：
- Critic 训练本质上是**价值函数拟合**
- 不需要专门的 Worker 类
- `TrainingWorker` 是通用训练引擎，可以处理任何模型训练任务

---

### 3.3 Megatron 策略处理

```python
elif config.critic.strategy == "megatron":
    # TODO: switch this to TrainingWorker as well
    from verl.workers.megatron_workers import CriticWorker
```

**注意：**
- Megatron 策略目前**不支持新版引擎**
- 必须使用 `CriticWorker` from `megatron_workers.py`
- TODO 注释表明：未来会迁移到 TrainingWorker

---

### 3.4 错误处理

```python
else:
    raise NotImplementedError
```

不支持其他策略，抛出 `NotImplementedError`。

---

### 3.5 注册映射

```python
from verl.trainer.ppo.ray_trainer import Role

self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
self.mapping[Role.Critic] = "global_pool"
```

与 ActorRollout 相同的注册模式：
- `role_worker_mapping`: Role → Worker 类映射
- `mapping`: Role → 资源池映射（固定为 `"global_pool"`）

---

## 四、Critic Worker 的作用

### 4.1 PPO 中 Critic 的角色

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO 算法中的 Critic                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入：状态 (State/Prompt + Response)                       │
│  输出：价值估计 V(s)                                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 价值函数 V(s)                         │   │
│  │                                                      │   │
│  │  V(s) = 期望的未来累积奖励                            │   │
│  │                                                      │   │
│  │  用于计算：                                           │   │
│  │  1. Advantage = A(s,a) = Q(s,a) - V(s)              │   │
│  │  2. GAE (Generalized Advantage Estimation)          │   │
│  │  3. Value Loss = MSE(V_pred, V_target)              │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  训练目标：                                                  │
│  - 最小化价值预测误差                                        │
│  - 提供稳定的基准线，减少策略更新方差                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 CriticWorker 的核心方法

**旧版 CriticWorker (fsdp_workers.py)：**

```python
class CriticWorker(FSDPWorker):
    
    @register(dispatch_mode=...)
    def compute_value(self, data: DataProto):
        """计算价值估计 V(s)"""
        pass
    
    @register(dispatch_mode=...)
    def update_critic(self, data: DataProto):
        """更新 Critic 模型参数"""
        pass
```

**新版 TrainingWorker (engine_workers.py)：**

```python
class TrainingWorker(Worker):
    
    @register(dispatch_mode=...)
    def train_model(self, data: DataProto):
        """通用训练方法，可用于 Critic 训练"""
        pass
    
    @register(dispatch_mode=...)
    def compute_value(self, data: DataProto):
        """计算价值估计"""
        pass
```

---

## 五、与 ActorRollout 的对比

| 方面 | ActorRollout Worker | Critic Worker |
|------|---------------------|---------------|
| **Role** | ActorRollout / ActorRolloutRef | Critic |
| **任务** | 生成rollout + Actor训练 | Critic训练 + 价值计算 |
| **资源池** | global_pool | global_pool |
| **新版引擎** | ActorRolloutRefWorker | TrainingWorker |
| **旧版Worker** | AsyncActorRolloutRefWorker | CriticWorker |
| **Megatron** | AsyncActorRolloutRefWorker | CriticWorker |

---

## 六、Worker 类详解

### 6.1 CriticWorker (旧版 fsdp)

**位置：** `verl/workers/fsdp_workers.py`

**继承关系：**
```python
class CriticWorker(FSDPWorker):
    """FSDP-based Critic Worker"""
```

**特点：**
- 基于 FSDP (Fully Sharded Data Parallel)
- 专门优化 Critic 训练流程
- 支持分布式价值计算

### 6.2 TrainingWorker (新版 engine)

**位置：** `verl/workers/engine_workers.py`

**继承关系：**
```python
class TrainingWorker(Worker):
    """通用训练 Worker，基于新引擎架构"""
```

**特点：**
- 不区分 Actor/Critic
- 统一的训练接口
- 更灵活的模型管理
- 内存效率更高（共享底层架构）

---

## 七、完整执行流程

```
add_critic_worker(config)
│
│  读取配置
│  ─────────────────────────────────────────────────────────────
│  use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
│  strategy = config.critic.strategy
│
│  选择Worker类
│  ─────────────────────────────────────────────────────────────
│  if strategy in {"fsdp", "fsdp2"}:
│      │
│      │ FSDP策略
│      │ ─────────────────────────────────────────────────────
│      │ if use_legacy_worker_impl in ["auto", "enable"]:
│      │     from verl.workers.fsdp_workers import CriticWorker
│      │     worker_cls = CriticWorker
│      │
│      │ elif use_legacy_worker_impl == "disable":
│      │     from verl.workers.engine_workers import TrainingWorker
│      │     worker_cls = TrainingWorker  # 新版通用Worker
│      │
│  elif strategy == "megatron":
│      │ Megatron策略（仅旧版）
│      │ ─────────────────────────────────────────────────────
│      │ from verl.workers.megatron_workers import CriticWorker
│      │ worker_cls = CriticWorker
│      │
│  else:
│      raise NotImplementedError
│
│  注册映射
│  ─────────────────────────────────────────────────────────────
│  self.role_worker_mapping[Role.Critic] = ray.remote(worker_cls)
│  self.mapping[Role.Critic] = "global_pool"
│
```

---

## 八、调用时机

在 `TaskRunner.run()` 中调用：

```python
def run(self, config):
    # 第一步：添加ActorRolloutWorker
    actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
    
    # 第二步：添加CriticWorker
    self.add_critic_worker(config)  # ← 这里调用
    
    # 第三步：添加其他Workers
    self.add_reward_model_worker(config)
    self.add_ref_policy_worker(config, actor_rollout_cls)
    
    # ...
```

---

## 九、设计差异分析

### 9.1 为什么 Critic 没有新版 Megatron 支持？

```
┌─────────────────────────────────────────────────────────────┐
│                  Megatron 策略的现状                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ActorRollout:                                              │
│  ───────────                                                │
│  ✅ 支持 AsyncActorRolloutRefWorker                         │
│  ❌ 不支持新版引擎 (use_legacy_worker_impl="disable")        │
│                                                             │
│  Critic:                                                    │
│  ───────                                                    │
│  ✅ 支持 CriticWorker                                       │
│  ❌ 不支持新版引擎 (TODO: switch to TrainingWorker)          │
│                                                             │
│  原因：                                                      │
│  ─────                                                      │
│  Megatron-LM 有特殊的分布式推理架构                          │
│  新版引擎 (SGLang/vLLM集成) 尚未适配                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 为什么 Critic 固定使用 global_pool？

```python
self.mapping[Role.Critic] = "global_pool"
```

**原因：**
- Critic 训练与 Actor 训练需要**共享同一个模型空间**
- Critic 计算的价值估计用于 Actor 的 Advantage 计算
- 两者需要紧密协作，放在同一资源池更高效

---

## 十、总结

| 层级 | 内容 |
|------|------|
| **输入** | config 配置对象 |
| **决策** | strategy (fsdp/fsdp2/megatron) + use_legacy_worker_impl |
| **输出** | Worker类注册到 role_worker_mapping |
| **资源** | 固定分配到 global_pool |

**一句话总结：**
根据策略选择 Critic Worker 类（FSDP用CriticWorker/TrainingWorker，Megatron用CriticWorker），注册到 Role.Critic 映射并分配到 global_pool 资源池。

**关键差异：**
- 新版引擎下，Critic 使用通用 TrainingWorker，不需要专用类
- Megatron 策略目前仅支持旧版 CriticWorker