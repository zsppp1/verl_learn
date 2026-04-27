# add_actor_rollout_worker 函数解析

> verl/trainer/main_ppo.py 第123-165行

## 一、函数定义

```python
def add_actor_rollout_worker(self, config):
    """Add actor rollout worker based on the actor strategy."""
```

**核心职责：**
- 根据配置选择合适的 Worker 类
- 创建 Role → Worker 的映射关系
- 返回 Worker 类和 WorkerGroup 类供后续使用

---

## 二、决策流程图

```
                    ┌─────────────────────────────────┐
                    │  use_legacy_worker_impl = ?     │
                    └─────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                "disable"     "auto"      "enable"
                    │            │            │
                    ▼            ▼            ▼
           ┌────────────┐ ┌────────────┐ ┌────────────┐
           │ 新版引擎    │ │ 旧版Worker │ │ 旧版Worker │
           │ Engine     │ │ Async      │ │ Async      │
           │ Worker     │ │ Worker     │ │ Worker     │
           └────────────┘ └────────────┘ └────────────┘
                    │            │            │
                    │            │            │
                    │    ┌───────┴───────┐    │
                    │    │ strategy = ?  │    │
                    │    └───────────────┘    │
                    │         │               │
                    │    ┌────┼────┐          │
                    │    │    │    │          │
                    │ "fsdp" "fsdp2" "megatron"
                    │    │    │    │          │
                    │    ▼    ▼    ▼          │
                    │ ┌────┐┌────┐┌────┐     │
                    │ │FSDP││FSDP││Mega│     │
                    │ │Work││Work││tron│     │
                    │ │er  ││er  ││Work│     │
                    │ └────┘└────┘│er  │     │
                    │            └────┘     │
                    │                         │
                    ▼                         ▼
        ┌──────────────────────────────────────────┐
        │  注册到 role_worker_mapping              │
        │  Role.ActorRollout → ray.remote(Worker)  │
        │  mapping[Role] = "global_pool"           │
        └──────────────────────────────────────────┘
```

---

## 三、详细代码解析

### 3.1 导入依赖

```python
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import Role
```

| 模块 | 作用 |
|------|------|
| `RayWorkerGroup` | 管理 Ray Worker 集合的类 |
| `Role` | 角色枚举，定义 Worker 类型 |

**Role枚举定义（ray_trainer.py）：**
```python
class Role(Enum):
    ActorRollout = "actor_rollout"        # Actor + Rollout
    ActorRolloutRef = "actor_rollout_ref" # Actor + Rollout + Reference
    Critic = "critic"
    RewardModel = "reward_model"
    RefPolicy = "ref_policy"
```

---

### 3.2 判断 Worker 实现版本

```python
use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
```

**配置值含义：**

| 值 | 含义 | Worker来源 |
|----|------|-----------|
| `"disable"` | 使用新版引擎 | `engine_workers.py` |
| `"auto"` | 自动选择（默认） | 根据strategy决定 |
| `"enable"` | 强制使用旧版 | `fsdp_workers.py` / `megatron_workers.py` |

---

### 3.3 新版引擎实现 (use_legacy_worker_impl == "disable")

```python
if use_legacy_worker_impl == "disable":
    from verl.workers.engine_workers import ActorRolloutRefWorker
    
    actor_rollout_cls = ActorRolloutRefWorker
    ray_worker_group_cls = RayWorkerGroup
    
    # 判断是否需要Reference Policy
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role = Role.ActorRolloutRef   # 包含Reference
    else:
        role = Role.ActorRollout      # 不包含Reference
    
    self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
    self.mapping[role] = "global_pool"
    return actor_rollout_cls, ray_worker_group_cls
```

**新版引擎特点：**

| 特点 | 说明 |
|------|------|
| 统一Worker | `ActorRolloutRefWorker` 同时处理 Actor + Rollout + Reference |
| Role动态选择 | 根据是否需要KL决定Role类型 |
| 内存优化 | Reference Policy与Actor共享，减少显存占用 |

**Role选择逻辑：**

```
┌─────────────────────────────────────────────────────────────┐
│         use_kl_in_reward / use_kl_loss = ?                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   True                      False                           │
│    │                         │                              │
│    ▼                         ▼                              │
│ ┌────────────────┐    ┌────────────────┐                    │
│ │ActorRolloutRef │    │ ActorRollout  │                    │
│ │ + Reference    │    │ (无Reference) │                    │
│ │ Policy         │    │               │                    │
│ └────────────────┘    └────────────────┘                    │
│                                                             │
│ 为什么分开？                                                 │
│ - 需要KL时，必须计算 π_ref 的 log_prob                      │
│ - 不需要KL时，省去Reference Policy，节省显存                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.4 旧版Worker实现 (use_legacy_worker_impl != "disable")

```python
# strategy选择：fsdp / fsdp2 / megatron
if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    actor_rollout_cls = AsyncActorRolloutRefWorker

elif config.actor_rollout_ref.actor.strategy == "megatron":
    from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
    actor_rollout_cls = AsyncActorRolloutRefWorker

else:
    raise NotImplementedError  # 不支持的策略

# 注册到映射
self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
self.mapping[Role.ActorRollout] = "global_pool"
return actor_rollout_cls, ray_worker_group_cls
```

**策略选择表：**

| strategy | Worker类 | 文件位置 | 特点 |
|----------|----------|----------|------|
| `"fsdp"` | AsyncActorRolloutRefWorker | fsdp_workers.py:1764 | Fully Sharded Data Parallel |
| `"fsdp2"` | AsyncActorRolloutRefWorker | fsdp_workers.py:1764 | FSDP v2 (更高效) |
| `"megatron"` | AsyncActorRolloutRefWorker | megatron_workers.py | Megatron-LM 分布式 |

---

## 四、Worker类详解

### 4.1 ActorRolloutRefWorker (新版)

**位置：** `verl/workers/engine_workers.py`

**功能：**
- Actor 训练（策略更新）
- Rollout 生成（vLLM/SGLang推理）
- Reference Policy（KL计算，可选）

**核心方法：**

```python
class ActorRolloutRefWorker(Worker):
    
    @register(dispatch_mode=...)
    def generate_sequences(self, prompts: DataProto):
        """生成rollout序列"""
        pass
    
    @register(dispatch_mode=...)
    def update_actor(self, data: DataProto):
        """更新Actor策略"""
        pass
    
    @register(dispatch_mode=...)
    def compute_log_prob(self, data: DataProto):
        """计算log_prob（用于KL）"""
        pass
```

### 4.2 AsyncActorRolloutRefWorker (旧版)

**位置：** `verl/workers/fsdp_workers.py:1764`

**继承关系：**
```python
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    """异步版本的ActorRolloutRefWorker"""
    pass
```

**特点：**
- 支持异步Rollout生成
- Actor训练与Rollout生成并行
- 更高吞吐量

---

## 五、注册映射详解

### 5.1 role_worker_mapping

```python
self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
```

**含义：**

```
┌───────────────────────────────────────────────────────────┐
│              role_worker_mapping 结构                      │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Key (Role)              Value (Ray Actor Class)          │
│  ─────────────────────   ───────────────────────────────  │
│  Role.ActorRollout   →   ray.remote(AsyncActorRolloutRef) │
│  Role.ActorRolloutRef →  ray.remote(ActorRolloutRefWorker)│
│                                                           │
│  用途：                                                   │
│  - RayPPOTrainer 根据 Role 找到对应的 Worker 类           │
│  - 初始化时创建 Worker 实例                               │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 5.2 mapping

```python
self.mapping[Role.ActorRollout] = "global_pool"
```

**含义：**

```
┌───────────────────────────────────────────────────────────┐
│                  mapping 结构                              │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Key (Role)              Value (Resource Pool ID)          │
│  ─────────────────────   ───────────────────────────────  │
│  Role.ActorRollout   →   "global_pool"                   │
│  Role.Critic         →   "global_pool"                   │
│  Role.RewardModel    →   "reward_pool" (可选独立池)       │
│                                                           │
│  用途：                                                   │
│  - 指定 Worker 在哪个资源池运行                           │
│  - ResourcePoolManager 根据此分配GPU                      │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

## 六、新旧版本对比

### 6.1 Architecture对比

```
┌─────────────────────────────────────────────────────────────────────┐
│              新版引擎 (use_legacy_worker_impl="disable")              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  ActorRolloutRefWorker                       │    │
│  │                                                              │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │    │
│  │   │   Actor     │  │   Rollout   │  │  Reference  │         │    │
│  │   │  (训练)     │  │  (生成)     │  │  Policy     │         │    │
│  │   └─────────────┘  └─────────────┘  └─────────────┘         │    │
│  │                                                              │    │
│  │   统一Worker类，共享权重                                      │    │
│  │   内存效率高                                                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              旧版Worker (use_legacy_worker_impl="auto/enable")       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────┐    ┌─────────────────────────┐         │
│  │  AsyncActorRolloutRef   │    │   RefPolicyWorker       │         │
│  │  Worker                 │    │   (可选，需要KL时)       │         │
│  │                         │    │                         │         │
│  │  ┌─────────┐┌─────────┐ │    │  ┌─────────────────┐   │         │
│  │  │ Actor   ││ Rollout  │ │    │  │ Reference Policy │   │         │
│  │  │         ││          │ │    │  │    (冻结模型)     │   │         │
│  │  └─────────┘└─────────┘ │    │  └─────────────────┘   │         │
│  │                         │    │                         │         │
│  └─────────────────────────┘    └─────────────────────────┘         │
│                                                                      │
│  Reference Policy 是独立的Worker                                    │
│  需要额外显存                                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Reference Policy处理对比

| 版本 | Reference Policy | Role | 显存 |
|------|------------------|------|------|
| 新版 | 融合在ActorRolloutRefWorker | ActorRolloutRef | 共享，省显存 |
| 旧版 | 独立RefPolicyWorker | RefPolicy | 需要额外加载 |

**新版代码注释原文：**
> "In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker, while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker."

---

## 七、完整执行流程

```
add_actor_rollout_worker(config)
│
│  读取配置
│  ─────────────────────────────────────────────────────────────
│  use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
│  strategy = config.actor_rollout_ref.actor.strategy
│  use_kl = config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss
│
│  选择Worker类
│  ─────────────────────────────────────────────────────────────
│  if use_legacy_worker_impl == "disable":
│      │
│      │ 新版引擎
│      │ ─────────────────────────────────────────────────────
│      │ from verl.workers.engine_workers import ActorRolloutRefWorker
│      │ actor_rollout_cls = ActorRolloutRefWorker
│      │
│      │ 选择Role
│      │ ─────────────────────────────────────────────────────
│      │ if use_kl:
│      │     role = Role.ActorRolloutRef   # 包含Reference
│      │ else:
│      │     role = Role.ActorRollout      # 不含Reference
│      │
│  else:
│      │ 旧版Worker
│      │ ─────────────────────────────────────────────────────
│      │ if strategy in {"fsdp", "fsdp2"}:
│      │     from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
│      │     actor_rollout_cls = AsyncActorRolloutRefWorker
│      │
│      │ elif strategy == "megatron":
│      │     from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
│      │     actor_rollout_cls = AsyncActorRolloutRefWorker
│      │
│      │ role = Role.ActorRollout  # 旧版固定用此Role
│      │
│  注册映射
│  ─────────────────────────────────────────────────────────────
│  self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
│  self.mapping[role] = "global_pool"
│
│  返回结果
│  ─────────────────────────────────────────────────────────────
│  return actor_rollout_cls, RayWorkerGroup
│
```

---

## 八、调用时机

在 `TaskRunner.run()` 中调用：

```python
def run(self, config):
    # 第一步：添加ActorRolloutWorker
    actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
    
    # 第二步：添加其他Workers
    self.add_critic_worker(config)
    self.add_reward_model_worker(config)
    self.add_ref_policy_worker(config, actor_rollout_cls)  # 使用返回的类
    
    # ...
    
    # 创建Trainer
    trainer = RayPPOTrainer(
        role_worker_mapping=self.role_worker_mapping,  # 传入映射
        ray_worker_group_cls=ray_worker_group_cls,     # 传入Group类
        ...
    )
```

---

## 九、总结

| 层级 | 内容 |
|------|------|
| **输入** | config 配置对象 |
| **决策** | use_legacy_worker_impl + strategy + use_kl |
| **输出** | Worker类 + Role映射 + WorkerGroup类 |
| **核心映射** | `Role → ray.remote(Worker)` |

**一句话总结：**
根据配置选择合适的 Actor Worker 类（新版engine或旧版fsdp/megatron），并根据是否需要KL惩罚决定Role类型，然后将Role与Worker的映射注册到 `role_worker_mapping` 中。