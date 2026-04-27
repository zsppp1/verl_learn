# add_ref_policy_worker 函数解析

> verl/trainer/main_ppo.py 第242-254行

## 一、函数定义

```python
def add_ref_policy_worker(self, config, ref_policy_cls):
    """Add reference policy worker if KL loss or KL reward is used."""
```

**核心职责：**
- 检查是否使用新版引擎（新版引擎不需要单独添加）
- 检查是否需要 KL 约束（KL in reward 或 KL loss）
- 添加 Reference Policy Worker（仅旧版引擎）

---

## 二、决策流程图

```
                ┌─────────────────────────────────┐
                │ use_legacy_worker_impl = ?      │
                └─────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
          "disable"                   "auto"/"enable"
              │                             │
              ▼                             ▼
    ┌──────────────────┐          ┌─────────────────────┐
    │ 新版引擎          │          │ 旧版 Worker         │
    │                  │          │                     │
    │ Ref Policy 已融合 │          │ 继续检查 KL 条件    │
    │ 在 ActorRollout  │          │                     │
    │ RefWorker 中     │          └─────────────────────┘
    │                  │                       │
    │ 直接 return      │                       ▼
    │ (不需要单独添加)  │          ┌─────────────────────────┐
    └──────────────────┘          │ use_kl_in_reward /      │
                                  │ use_kl_loss = ?         │
                                  └─────────────────────────┘
                                              │
                                    ┌─────────┼─────────┐
                                    │                   │
                                   True               False
                                    │                   │
                                    ▼                   ▼
                            ┌────────────────┐  ┌─────────────────┐
                            │ 添加 RefPolicy │  │ 不添加 Worker   │
                            │ Worker         │  │ (函数结束)       │
                            │                │  │                 │
                            │ mapping →      │  │ 不需要 KL 约束   │
                            │ global_pool    │  │                 │
                            └────────────────┘  └─────────────────┘
```

---

## 三、详细代码解析

### 3.1 新版引擎判断

```python
# Ref policy has been fused into ActorRolloutRefWorker in new model engine,
# we don't need to add a separate ref policy worker group.
use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
if use_legacy_worker_impl == "disable":
    return
```

**源码注释解读：**

> "Ref policy has been fused into ActorRolloutRefWorker in new model engine, we don't need to add a separate ref policy worker group."

**翻译：**
> "在新版模型引擎中，Reference Policy 已经融合到 ActorRolloutRefWorker 中，我们不需要添加单独的 reference policy worker group。"

**新旧版本对比：**

```
┌─────────────────────────────────────────────────────────────────────┐
│              Reference Policy 在新旧版本中的处理方式                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  旧版 Worker (use_legacy_worker_impl = "auto" / "enable"):          │
│  ───────────────────────────────────────────────────────────        │
│                                                                      │
│  ┌──────────────────────┐      ┌──────────────────────┐            │
│  │ ActorRollout Worker  │      │ RefPolicy Worker     │            │
│  │                      │      │ (独立 Worker)         │            │
│  │ ┌─────────┐┌────────┐│      │ ┌─────────────────┐  │            │
│  │ │ Actor   ││ Rollout ││      │ │ Reference Model │  │            │
│  │ │ (训练)  ││ (生成)  ││      │ │    (冻结)       │  │            │
│  │ └─────────┘└────────┘│      │ └─────────────────┘  │            │
│  │                      │      │                      │            │
│  │ 需要显存加载 Actor    │      │ 需要额外显存加载 Ref │            │
│  │                      │      │                      │            │
│  └──────────────────────┘      └──────────────────────┘            │
│                                                                      │
│  缺点：                                                              │
│  ❌ 两个独立 Worker，需要两次加载模型                                 │
│  ❌ 占用更多显存                                                     │
│  ❌ Actor 和 Ref Policy 不共享权重                                   │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  新版引擎 (use_legacy_worker_impl = "disable"):                     │
│  ───────────────────────────────────────────────────────────        │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  ActorRolloutRefWorker                        │  │
│  │                                                               │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │  │
│  │   │   Actor     │  │   Rollout   │  │  Reference  │          │  │
│  │   │  (训练)     │  │  (生成)     │  │   Policy    │          │  │
│  │   └─────────────┘  └─────────────┘  └─────────────┘          │  │
│  │                                                               │  │
│  │   三者融合在一个 Worker 中                                     │  │
│  │                                                               │  │
│  │   ┌──────────────────────────────────────────────────────┐  │  │
│  │   │   Actor 和 Reference Policy 共享模型权重               │  │  │
│  │   │   只需要一次加载，节省显存                              │  │  │
│  │   │   Reference Policy 临时切换为冻结状态计算 log_prob     │  │  │
│  │   └──────────────────────────────────────────────────────┘  │  │
│  │                                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  优点：                                                              │
│  ✅ 单个 Worker，一次性加载                                          │
│  ✅ 节省显存（共享权重）                                              │
│  ✅ 无需单独添加 RefPolicy Worker                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 KL 条件检查

```python
if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
    self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
    self.mapping[Role.RefPolicy] = "global_pool"
```

**两种 KL 用途：**

| 用途 | 配置项 | 作用 |
|------|--------|------|
| **KL in Reward** | `config.algorithm.use_kl_in_reward` | KL 散度作为奖励惩罚项 |
| **KL Loss** | `config.actor_rollout_ref.actor.use_kl_loss` | KL 散度作为训练损失项 |

---

## 四、Reference Policy 的作用

### 4.1 KL 散度约束

```
┌─────────────────────────────────────────────────────────────────────┐
│                  KL 散度在 PPO 中的作用                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  为什么需要 KL 散度约束？                                             │
│  ───────────────────────                                             │
│                                                                      │
│  PPO 目标：                                                          │
│  max L(θ) = E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ]            │
│                                                                      │
│  问题：                                                              │
│  - 如果 π_new 与 π_old 差距太大，策略更新不稳定                       │
│  - 可能导致性能下降或震荡                                             │
│                                                                      │
│  解决：                                                              │
│  - 引入 KL 散度约束，限制策略更新幅度                                 │
│  - KL(π_new || π_ref) < δ                                           │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   KL 散度公式：                                                │  │
│  │                                                               │  │
│  │   KL(π_new || π_ref) = Σ π_new(x) * log(π_new(x) / π_ref(x)) │  │
│  │                                                               │  │
│  │   表示：新策略 π_new 与参考策略 π_ref 的差异程度              │  │
│  │                                                               │  │
│  │   π_ref = π_initial (初始策略，冻结不变)                      │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 KL in Reward vs KL Loss

```
┌─────────────────────────────────────────────────────────────────────┐
│              KL in Reward 与 KL Loss 的区别                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  KL in Reward:                                                       │
│  ─────────────                                                       │
│                                                                      │
│  将 KL 散度作为奖励惩罚项                                             │
│                                                                      │
│  reward_final = reward - β * KL(π_new || π_ref)                     │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  意义：                                                       │  │
│  │  - 如果新策略偏离参考策略太多，奖励会被惩罚                    │  │
│  │  - Actor 会自然地学习不要偏离太远                              │  │
│  │                                                               │  │
│  │  β 是 KL 惩罚系数，控制惩罚强度                                │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  KL Loss:                                                            │
│  ────────                                                            │
│                                                                      │
│  将 KL 散度作为训练损失项                                             │
│                                                                      │
│  L_total = L_ppo + α * KL(π_new || π_ref)                           │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  意义：                                                       │  │
│  │  - KL 散度直接参与梯度计算                                    │  │
│  │  - 直接约束策略更新方向                                       │  │
│  │                                                               │  │
│  │  α 是 KL 损失系数，控制约束强度                                │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  选择：                                                              │
│  ─────                                                              │
│  - KL in Reward: 更平滑，适合探索                                    │
│  - KL Loss: 更直接，适合稳定性                                       │
│  - 可以两者同时使用                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 4.3 ref_policy_cls 参数

```python
def add_ref_policy_worker(self, config, ref_policy_cls):
```

**ref_policy_cls 来自哪里？**

```python
# TaskRunner.run() 中调用
actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
self.add_ref_policy_worker(config, actor_rollout_cls)  # ← actor_rollout_cls 作为 ref_policy_cls
```

**为什么 Actor Rollout Worker 类可以作为 Ref Policy Worker？**

```
┌─────────────────────────────────────────────────────────────────────┐
│           Actor Rollout Worker 如何作为 Ref Policy Worker             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  AsyncActorRolloutRefWorker 结构：                                   │
│  ────────────────────────────────                                    │
│                                                                      │
│  class AsyncActorRolloutRefWorker:                                  │
│      def __init__(self):                                            │
│          self.actor_model = ...      # Actor 模型                   │
│          self.ref_policy_model = ... # Reference Policy 模型        │
│                                                                      │
│      @register                                                      │
│      def compute_ref_log_prob(self, data):                          │
│          """计算 Reference Policy 的 log_prob"""                    │
│          with torch.no_grad():  # 冻结，不计算梯度                   │
│              return self.ref_policy_model.compute_log_prob(data)    │
│                                                                      │
│  特点：                                                              │
│  ─────                                                              │
│  1. 同一个 Worker 类可以加载多个模型                                 │
│  2. Reference Policy 是 Actor 模型的冻结副本                         │
│  3. 代码复用：同一个类处理 Actor 和 Ref Policy                       │
│                                                                      │
│  配置差异：                                                          │
│  ─────────                                                          │
│  - Actor Worker: 模型可训练                                         │
│  - Ref Policy Worker: 模型冻结，torch.no_grad()                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 五、与 ActorRollout 的关系

### 5.1 新版引擎的 Role 选择回顾

```python
# add_actor_rollout_worker 中
if use_legacy_worker_impl == "disable":
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role = Role.ActorRolloutRef   # 包含 Reference
    else:
        role = Role.ActorRollout      # 不包含 Reference
```

**Role 的含义：**

| Role | 包含 | 使用场景 |
|------|------|----------|
| `Role.ActorRollout` | Actor + Rollout | 不需要 KL 约束 |
| `Role.ActorRolloutRef` | Actor + Rollout + Reference | 需要 KL 约束 |

### 5.2 新旧版本的 Role 差异

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Role 在新旧版本中的差异                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  新版引擎 (use_legacy_worker_impl = "disable"):                     │
│  ───────────────────────────────────────────────────────────        │
│                                                                      │
│  需要 KL:                                                            │
│  role_worker_mapping[ActorRolloutRef] = ActorRolloutRefWorker      │
│  (Reference Policy 已融合，不需要单独 Role.RefPolicy)               │
│                                                                      │
│  不需要 KL:                                                          │
│  role_worker_mapping[ActorRollout] = ActorRolloutRefWorker         │
│  (Worker 类相同，但 Role 不同)                                       │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  旧版 Worker (use_legacy_worker_impl = "auto" / "enable"):          │
│  ───────────────────────────────────────────────────────────        │
│                                                                      │
│  Actor/Rollout:                                                      │
│  role_worker_mapping[ActorRollout] = AsyncActorRolloutRefWorker    │
│                                                                      │
│  Ref Policy (需要 KL 时):                                            │
│  role_worker_mapping[RefPolicy] = AsyncActorRolloutRefWorker       │
│  (同一个类，但不同的 Role 和配置)                                    │
│                                                                      │
│  映射示例：                                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  Role.ActorRollout  → AsyncActorRolloutRefWorker           │   │
│  │  Role.RefPolicy     → AsyncActorRolloutRefWorker           │   │
│  │                                                             │   │
│  │  RayPPOTrainer 根据不同的 Role 初始化不同的配置：            │   │
│  │  - ActorRollout: Actor 可训练                               │   │
│  │  - RefPolicy: Reference Policy 冻结                         │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、完整执行流程

```
add_ref_policy_worker(config, ref_policy_cls)
│
│  检查新版引擎
│  ─────────────────────────────────────────────────────────────
│  use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
│  
│  if use_legacy_worker_impl == "disable":
│      │
│      │ 新版引擎：Ref Policy 已融合在 ActorRolloutRefWorker
│      │ ─────────────────────────────────────────────────────
│      │ 不需要单独添加 Ref Policy Worker
│      │
│      return  # 函数结束
│
│  检查 KL 条件
│  ─────────────────────────────────────────────────────────────
│  if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
│      │
│      │ 需要 KL 约束：添加 Ref Policy Worker
│      │ ─────────────────────────────────────────────────────
│      │ self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
│      │ self.mapping[Role.RefPolicy] = "global_pool"
│      │
│  else:
│      │ 不需要 KL 约束：不添加
│      │ ─────────────────────────────────────────────────────
│      │ 函数结束
│
```

---

## 七、调用时机

在 `TaskRunner.run()` 中调用：

```python
def run(self, config):
    # add_actor_rollout_worker 返回 actor_rollout_cls
    actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
    
    self.add_critic_worker(config)
    self.add_reward_model_worker(config)
    
    # 传入 actor_rollout_cls 作为 ref_policy_cls ← 这里调用
    self.add_ref_policy_worker(config, actor_rollout_cls)
    
    # 初始化资源池
    resource_pool_manager = self.init_resource_pool_mgr(config)
```

**为什么传入 actor_rollout_cls？**
- Actor 和 Ref Policy 使用同一个模型
- 同一个 Worker 类可以处理两者
- 配置差异在 RayPPOTrainer.init_workers() 中处理

---

## 八、总结

| 层级 | 内容 |
|------|------|
| **输入** | config + ref_policy_cls (来自 add_actor_rollout_worker) |
| **决策** | use_legacy_worker_impl → use_kl |
| **输出** | RefPolicy Worker 注册（仅旧版且需要KL时） |
| **资源** | 固定分配到 global_pool |

**一句话总结：**
新版引擎不需要单独添加 Ref Policy Worker（已融合在 ActorRolloutRefWorker）；旧版引擎仅在需要 KL 约束时添加，使用与 Actor 相同的 Worker 类但不同配置。

**关键设计：**
- 新版引擎：Actor + Rollout + Ref Policy 三合一
- 旧版引擎：Actor/Rollout 与 Ref Policy 分开
- 都使用 global_pool（因为需要共享模型权重空间）