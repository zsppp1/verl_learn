# add_reward_model_worker 函数解析

> verl/trainer/main_ppo.py 第216-240行

## 一、函数定义

```python
def add_reward_model_worker(self, config):
    """Add reward model worker if enabled."""
```

**核心职责：**
- 检查是否启用 Reward Model
- 根据策略选择 RewardModelWorker 类
- 注册 Role.RewardModel → Worker 的映射
- 根据配置选择资源池（reward_pool 或 global_pool）

---

## 二、决策流程图

```
                ┌─────────────────────────────────┐
                │   reward_model.enable = ?       │
                └─────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
            True                          False
              │                             │
              ▼                             ▼
    ┌──────────────────┐          ┌─────────────────┐
    │ 添加 RewardModel │          │ 不添加 Worker   │
    │ Worker           │          │ (函数结束)       │
    └──────────────────┘          └─────────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │   reward_model.strategy = ?     │
    └─────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │                   │
  "fsdp/fsdp2"        "megatron"
    │                   │
    ▼                   ▼
┌────────────────┐ ┌────────────────┐
│ FSDP           │ │ Megatron       │
│ RewardModel    │ │ RewardModel    │
│ Worker         │ │ Worker         │
└────────────────┘ └────────────────┘
              │
              ▼
    ┌─────────────────────────────────┐
    │ enable_resource_pool = ?        │
    └─────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │                   │
   True               False
    │                   │
    ▼                   ▼
┌────────────────┐ ┌────────────────┐
│ reward_pool    │ │ global_pool    │
│ (独立资源池)   │ │ (共享池)       │
└────────────────┘ └────────────────┘
```

---

## 三、详细代码解析

### 3.1 启用检查

```python
if config.reward_model.enable:
    # ... 添加 RewardModelWorker
```

**Reward Model 的作用：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Reward Model 在 RLHF 中的作用                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入：Prompt + Response                                             │
│  输出：Reward Score (奖励分数)                                        │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   Prompt: "请解释什么是PPO算法"                                │  │
│  │   Response: "PPO是Proximal Policy Optimization..."            │  │
│  │                                                               │  │
│  │   Reward Model 评分: 0.85                                     │  │
│  │                                                               │  │
│  │   评分标准：                                                   │  │
│  │   - 回答准确性                                                 │  │
│  │   - 语言流畅度                                                 │  │
│  │   - 内容有用性                                                 │  │
│  │   - 符合人类偏好                                               │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  不使用 Reward Model 的情况：                                         │
│  ─────────────────────────                                           │
│  - 使用规则奖励（如代码执行结果）                                     │
│  - 使用预定义的评分函数                                               │
│  - 由其他方式提供奖励信号                                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Worker 类选择

```python
use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
if use_legacy_worker_impl in ["auto", "enable", "disable"]:
    if config.reward_model.strategy in {"fsdp", "fsdp2"}:
        from verl.workers.fsdp_workers import RewardModelWorker
    elif config.reward_model.strategy == "megatron":
        from verl.workers.megatron_workers import RewardModelWorker
    else:
        raise NotImplementedError
```

**Worker 类来源表：**

| strategy | Worker类 | 文件位置 | 说明 |
|----------|----------|----------|------|
| `"fsdp"` / `"fsdp2"` | RewardModelWorker | fsdp_workers.py | FSDP 分布式推理 |
| `"megatron"` | RewardModelWorker | megatron_workers.py | Megatron-LM 分布式 |

**与 Actor/Critic 的差异：**

```
┌─────────────────────────────────────────────────────────────────────┐
│       Reward Model Worker 与 Actor/Critic Worker 的差异              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Actor/Critic:                                                       │
│  ─────────────                                                       │
│  新版引擎 (use_legacy_worker_impl="disable"):                        │
│  ✅ 使用 TrainingWorker / ActorRolloutRefWorker                     │
│  ✅ 有新版引擎支持                                                   │
│                                                                      │
│  Reward Model:                                                       │
│  ─────────────                                                       │
│  新版引擎 (use_legacy_worker_impl="disable"):                        │
│  ❌ 仍使用旧版 RewardModelWorker                                     │
│  ❌ 注释掉的新版代码（见下文）                                        │
│                                                                      │
│  原因：                                                              │
│  ─────                                                              │
│  Reward Model 只做推理，不需要训练                                   │
│  新版引擎主要优化训练流程                                            │
│  Reward Model 推理已有成熟方案                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.3 注释掉的新版代码

```python
# elif use_legacy_worker_impl == "disable":
#     from verl.workers.engine_workers import RewardModelWorker
#
#     print("Using new worker implementation")
```

**这说明：**
- 曾经考虑过 Reward Model 使用新版引擎
- 目前已注释掉，仍使用旧版
- 可能未来会启用

---

### 3.4 注册映射

```python
self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
```

与 Actor/Critic 相同的注册方式：
- `Role.RewardModel` → `ray.remote(RewardModelWorker)`
- 使用 Ray Actor 实现分布式

---

### 3.5 资源池选择

```python
if config.reward_model.enable_resource_pool:
    self.mapping[Role.RewardModel] = "reward_pool"
else:
    self.mapping[Role.RewardModel] = "global_pool"
```

**这是 Reward Model 独有的特性：**

```
┌─────────────────────────────────────────────────────────────────────┐
│              Reward Model 的资源池选择逻辑                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  enable_resource_pool = True:                                        │
│  ─────────────────────────────                                       │
│  │                                                                   │
│  │  mapping[Role.RewardModel] = "reward_pool"                       │
│  │                                                                   │
│  │  ┌───────────────────────────────────────────────────────────┐  │
│  │  │                reward_pool (独立GPU)                       │  │
│  │  │                                                            │  │
│  │  │   Reward Model Worker                                      │  │
│  │  │                                                            │  │
│  │  │   - 独立运行，不竞争 Actor/Critic 的 GPU                   │  │
│  │  │   - 可以并行计算 reward                                    │  │
│  │  │   - 适合大规模部署                                         │  │
│  │  │                                                            │  │
│  │  └───────────────────────────────────────────────────────────┘  │
│  │                                                                   │
│                                                                      │
│  enable_resource_pool = False:                                       │
│  ──────────────────────────────                                      │
│  │                                                                   │
│  │  mapping[Role.RewardModel] = "global_pool"                       │
│  │                                                                   │
│  │  ┌───────────────────────────────────────────────────────────┐  │
│  │  │                global_pool (共享GPU)                       │  │
│  │  │                                                            │  │
│  │  │   Actor │ Critic │ RefPolicy │ Reward Model               │  │
│  │  │                                                            │  │
│  │  │   所有 Worker 共享 GPU                                     │  │
│  │  │   需要协调调度                                             │  │
│  │  │                                                            │  │
│  │  └───────────────────────────────────────────────────────────┘  │
│  │                                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、RewardModelWorker 类详解

### 4.1 位置和功能

**位置：** `verl/workers/fsdp_workers.py` 或 `megatron_workers.py`

**核心功能：**

```python
class RewardModelWorker:
    
    @register(dispatch_mode=...)
    def compute_reward(self, data: DataProto):
        """计算 reward score"""
        # 输入: Prompt + Response
        # 输出: Reward Score
        pass
```

### 4.2 Reward Model 的训练方式

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Reward Model 的训练方式                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  训练阶段（PPO 之前）：                                               │
│  ──────────────────────                                              │
│                                                                      │
│  1. 收集人类偏好数据：                                                │
│     ┌─────────────────────────────────────────────────────────────┐ │
│     │ Prompt: "请解释什么是PPO"                                     │ │
│     │ Response A: "PPO是一种强化学习算法..."                        │ │
│     │ Response B: "PPO是..."                                       │ │
│     │ 人类选择: Response A 更好                                     │ │
│     └─────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  2. 训练 Reward Model：                                              │
│     模型学习预测人类偏好                                              │
│     输入: (Prompt, Response)                                         │
│     输出: 好坏程度评分                                                │
│                                                                      │
│  PPO 阶段：                                                          │
│  ─────────                                                           │
│                                                                      │
│  3. Reward Model 已冻结：                                            │
│     只做推理，不参与训练                                              │
│     为 Actor 生成的 response 打分                                    │
│                                                                      │
│     ┌───────────────────────────────────────────────────────────┐  │
│     │                                                             │  │
│     │   Actor 生成 Response → Reward Model → Score              │  │
│     │                                                             │  │
│     │   Score 用于计算 PPO 的奖励信号                              │  │
│     │                                                             │  │
│     └───────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 五、与其他 Worker 的对比

| 方面 | ActorRollout Worker | Critic Worker | RewardModel Worker |
|------|---------------------|---------------|-------------------|
| **Role** | ActorRollout / ActorRolloutRef | Critic | RewardModel |
| **任务** | 生成 + Actor训练 | Critic训练 | Reward推理 |
| **新版引擎** | ✅ ActorRolloutRefWorker | ✅ TrainingWorker | ❌ (注释掉) |
| **资源池** | global_pool | global_pool | reward_pool / global_pool |
| **是否训练** | ✅ Actor训练 | ✅ Critic训练 | ❌ 只推理 |
| **独立池选项** | ❌ | ❌ | ✅ |

---

## 六、配置示例

### 6.1 启用 Reward Model（共享池）

```yaml
reward_model:
  enable: true
  enable_resource_pool: false  # 使用 global_pool
  strategy: "fsdp"
```

**结果：**
```python
role_worker_mapping[Role.RewardModel] = RewardModelWorker
mapping[Role.RewardModel] = "global_pool"
```

### 6.2 启用 Reward Model（独立池）

```yaml
reward_model:
  enable: true
  enable_resource_pool: true  # 使用独立 reward_pool
  strategy: "fsdp"
  n_gpus_per_node: 4
  nnodes: 2
```

**结果：**
```python
role_worker_mapping[Role.RewardModel] = RewardModelWorker
mapping[Role.RewardModel] = "reward_pool"
# init_resource_pool_mgr 会创建 reward_pool: [4, 4]
```

### 6.3 不使用 Reward Model

```yaml
reward_model:
  enable: false
```

**结果：**
- 函数不执行任何操作
- 不添加 RewardModelWorker
- 使用其他方式提供奖励信号（如规则奖励）

---

## 七、独立资源池的优势

```
┌─────────────────────────────────────────────────────────────────────┐
│              独立 reward_pool 的优势                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  性能优势：                                                          │
│  ─────────                                                          │
│  ✅ Actor 训练和 Reward 计算可以并行                                 │
│  ✅ 避免 GPU 资源竞争                                                │
│  ✅ 提高整体吞吐量                                                   │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  时间线对比：                                                  │  │
│  │                                                               │  │
│  │  共享池 (global_pool):                                        │  │
│  │  ─────────────────────                                        │  │
│  │  Actor训练 ────┤ Reward计算 ────┤ Actor训练 ────┤ ...        │  │
│  │               (串行，等待)                                     │  │
│  │                                                               │  │
│  │  独立池 (reward_pool):                                        │  │
│  │  ─────────────────────                                        │  │
│  │  Actor训练 ────┤ Actor训练 ────┤ ...                         │  │
│  │  Reward计算 ────┤ Reward计算 ────┤ ...                       │  │
│  │               (并行，不等待)                                   │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  成本优势：                                                          │
│  ─────────                                                          │
│  ✅ Reward Model 可以使用更少/更便宜的 GPU                           │
│  ✅ 推理不需要高精度计算，可用低端GPU                                 │
│                                                                      │
│  扩展优势：                                                          │
│  ─────────                                                          │
│  ✅ 可以独立扩展 Reward Model 的规模                                 │
│  ✅ 支持 Reward Model 的分布式部署                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 八、完整执行流程

```
add_reward_model_worker(config)
│
│  检查启用状态
│  ─────────────────────────────────────────────────────────────
│  if config.reward_model.enable:
│      │ (继续执行)
│  else:
│      return  # 函数结束，不添加 Worker
│
│  选择 Worker 类
│  ─────────────────────────────────────────────────────────────
│  use_legacy_worker_impl = config.trainer.get(...)
│  
│  if use_legacy_worker_impl in ["auto", "enable", "disable"]:
│      │
│      │ if reward_model.strategy in {"fsdp", "fsdp2"}:
│      │     from verl.workers.fsdp_workers import RewardModelWorker
│      │
│      │ elif reward_model.strategy == "megatron":
│      │     from verl.workers.megatron_workers import RewardModelWorker
│      │
│      │ else:
│      │     raise NotImplementedError
│      │
│  else:
│      raise ValueError
│
│  注册映射
│  ─────────────────────────────────────────────────────────────
│  self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
│
│  选择资源池
│  ─────────────────────────────────────────────────────────────
│  if config.reward_model.enable_resource_pool:
│      self.mapping[Role.RewardModel] = "reward_pool"  # 独立池
│  else:
│      self.mapping[Role.RewardModel] = "global_pool"  # 共享池
│
```

---

## 九、调用时机

在 `TaskRunner.run()` 中调用：

```python
def run(self, config):
    actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
    self.add_critic_worker(config)
    
    # 添加 Reward Model Worker ← 这里调用
    self.add_reward_model_worker(config)
    
    self.add_ref_policy_worker(config, actor_rollout_cls)
    
    # 初始化资源池（会根据 mapping 创建 reward_pool）
    resource_pool_manager = self.init_resource_pool_mgr(config)
```

---

## 十、总结

| 层级 | 内容 |
|------|------|
| **输入** | config.reward_model (enable, strategy, enable_resource_pool) |
| **决策** | enable → strategy → resource_pool |
| **输出** | RewardModelWorker 注册到 role_worker_mapping |
| **资源** | reward_pool (独立) 或 global_pool (共享) |

**一句话总结：**
检查是否启用 Reward Model，根据策略选择 RewardModelWorker，并根据配置分配到独立 reward_pool 或共享 global_pool。

**关键差异：**
- Reward Model 是唯一支持独立资源池的 Worker
- 新版引擎目前不支持 Reward Model Worker（注释状态）
- Reward Model 只做推理，不参与 PPO 训练