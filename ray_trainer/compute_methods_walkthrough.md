# 计算辅助方法详解

> verl/trainer/ppo/ray_trainer.py 行1143-1240

## 一、方法概述

| 方法 | 行号 | 职责 |
|------|------|------|
| `_compute_values()` | 1143-1158 | 计算 Critic Value |
| `_compute_ref_log_prob()` | 1160-1179 | 计算 Reference Policy log_prob |
| `_compute_old_log_prob()` | 1181-1204 | 计算 Actor old_log_prob |
| `_update_actor()` | 1206-1240 | 更新 Actor 参数 |
| `_update_critic()` | 1242-1270 | 更新 Critic 参数 |

---

## 二、_compute_values 方法

### 2.1 方法定义

```python
def _compute_values(self, batch: DataProto) -> DataProto:
    """Compute value estimates using Critic WorkerGroup."""
```

**核心职责：**
- 调用 Critic WorkerGroup 计算 Value V(s)

### 2.2 代码详解

```python
def _compute_values(self, batch: DataProto) -> DataProto:
    if self.use_legacy_worker_impl == "disable":
        # 新版引擎处理流程
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
    else:
        # 旧版 Worker
        values = self.critic_wg.compute_values(batch)
    return values
```

### 2.3 新版引擎的数据转换流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                  新版引擎数据转换流程                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: DataProto → TensorDict                                    │
│  ───────────────────────────                                        │
│  batch_td = batch.to_tensordict()                                  │
│                                                                      │
│  Step 2: Padding → NoPadding                                       │
│  ──────────────────────────                                        │
│  batch_td = left_right_2_no_padding(batch_td)                      │
│                                                                      │
│  Padding 格式：                                                      │
│  [token_1, token_2, ..., PAD, PAD]                                 │
│                                                                      │
│  NoPadding 格式：                                                    │
│  [token_1, token_2, ...]  (去掉 PAD)                               │
│                                                                      │
│  原因：                                                              │
│  ─────                                                              │
│  新版引擎的 TrainingWorker 不需要 padding                          │
│  更高效的处理方式                                                   │
│                                                                      │
│  Step 3: 添加 meta_info                                             │
│  ─────────────────────                                              │
│  tu.assign_non_tensor(batch_td, compute_loss=False)                │
│                                                                      │
│  Step 4: RPC 调用                                                   │
│  ─────────                                                          │
│  output = self.critic_wg.infer_batch(batch_td)                     │
│  output = output.get()  # 等待结果                                 │
│                                                                      │
│  Step 5: NoPadding → Padding                                       │
│  ──────────────────────────                                        │
│  values = no_padding_2_padding(values, batch_td)                   │
│                                                                      │
│  Step 6: TensorDict → DataProto                                    │
│  ───────────────────────────                                        │
│  values = DataProto.from_tensordict(values)                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、_compute_ref_log_prob 方法

### 3.1 方法定义

```python
def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
    """Compute log probabilities using Reference Policy WorkerGroup."""
```

**核心职责：**
- 调用 RefPolicy WorkerGroup 计算 π_ref 的 log_prob

### 3.2 代码详解

```python
def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
    if self.use_legacy_worker_impl == "disable":
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
        output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        log_probs = tu.get(output, "log_probs")
        log_probs = no_padding_2_padding(log_probs, batch_td)
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)
    else:
        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    return ref_log_prob
```

### 3.3 ref_log_prob 用途

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ref_log_prob 用途                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  KL 散度计算：                                                       │
│  ─────────                                                          │
│  KL = Σ π_new(a|s) × log(π_new(a|s) / π_ref(a|s))                  │
│                                                                      │
│  需要：                                                              │
│  - old_log_prob = log π_new(a|s)                                   │
│  - ref_log_prob = log π_ref(a|s)                                   │
│                                                                      │
│  计算：                                                              │
│  kld = old_log_probs - ref_log_prob                                │
│                                                                      │
│  或者：                                                              │
│  kld = exp(old_log_probs) × (old_log_probs - ref_log_prob)         │
│                                                                      │
│  简化：                                                              │
│  kld = π_new × (log π_new - log π_ref)                             │
│                                                                      │
│  用途：                                                              │
│  ─────                                                              │
│  1. KL in reward: token_level_rewards = scores - β × KL            │
│  2. KL loss: L_total = L_ppo + α × KL                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、_compute_old_log_prob 方法

### 4.1 方法定义

```python
def _compute_old_log_prob(self, batch: DataProto) -> tuple[DataProto, float]:
    """Compute log probabilities using Actor WorkerGroup, returns old_log_prob and MFU."""
```

**核心职责：**
- 调用 Actor WorkerGroup 计算 π_old 的 log_prob
- 同时计算 entropy 和 MFU

### 4.2 代码详解

```python
def _compute_old_log_prob(self, batch: DataProto):
    if self.use_legacy_worker_impl == "disable":
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        
        old_log_prob = tu.get_tensordict({
            "old_log_probs": log_probs.float(),
            "entropys": entropy.float()
        })
        old_log_prob = DataProto.from_tensordict(old_log_prob)
    else:
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        old_log_prob_mfu = 0
    return old_log_prob, old_log_prob_mfu
```

### 4.3 old_log_prob 用途

```
┌─────────────────────────────────────────────────────────────────────┐
│                  old_log_prob 用途                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PPO Importance Sampling：                                          │
│  ──────────────────────────                                         │
│                                                                      │
│  ratio = π_new(a|s) / π_old(a|s)                                   │
│                                                                      │
│  计算：                                                              │
│  ─────                                                              │
│  log_ratio = log π_new - log π_old                                 │
│  ratio = exp(log_ratio)                                            │
│                                                                      │
│  PPO Loss：                                                          │
│  ─────────                                                          │
│  L = min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)                     │
│                                                                      │
│  其中：                                                              │
│  - π_old = rollout 时策略的 log_prob                               │
│  - π_new = 更新后策略的 log_prob                                   │
│  - A = Advantage                                                   │
│  - ε = clip epsilon                                               │
│                                                                      │
│  entropy 用途：                                                      │
│  ─────────                                                          │
│  L_total = L_ppo - c × entropy                                    │
│                                                                      │
│  熵惩罚鼓励探索                                                      │
│                                                                      │
│  MFU (Model FLOPs Utilization)：                                    │
│  ─────────────────────────────                                      │
│  模型计算效率指标                                                    │
│  用于性能监控                                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 五、_update_actor 方法

### 5.1 方法定义

```python
def _update_actor(self, batch: DataProto) -> DataProto:
    """Update Actor policy using WorkerGroup."""
```

**核心职责：**
- 调用 Actor WorkerGroup 执行策略更新

### 5.2 代码详解

```python
def _update_actor(self, batch: DataProto) -> DataProto:
    rollout_config = self.config.actor_rollout_ref.rollout
    batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
    batch.meta_info["temperature"] = rollout_config.temperature
    
    if self.use_legacy_worker_impl == "disable":
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        
        calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            ...
        )
        
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
    else:
        actor_output = self.actor_rollout_wg.update_actor(batch)
    return actor_output
```

### 5.3 Actor 更新参数

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Actor 更新参数                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ppo_mini_batch_size:                                               │
│  ───────────────────                                                │
│  PPO mini-batch 大小                                                │
│  将整个 batch 分成多个 mini-batch 进行多次更新                     │
│                                                                      │
│  示例：                                                              │
│  batch_size = 256                                                   │
│  mini_batch_size = 64                                               │
│  → 4 个 mini-batch                                                  │
│                                                                      │
│  ppo_epochs:                                                        │
│  ─────────                                                          │
│  每个 batch 更新次数                                                │
│                                                                      │
│  示例：                                                              │
│  ppo_epochs = 4                                                     │
│  mini_batch_size = 64                                               │
│  → 每个样本被更新 4 次                                              │
│                                                                      │
│  entropy_coeff:                                                     │
│  ─────────────                                                      │
│  熵惩罚系数                                                          │
│  entropy_coeff > 0 → 鼓励探索                                      │
│                                                                      │
│  shuffle:                                                           │
│  ───────                                                            │
│  是否在 mini-batch 间 shuffle                                      │
│                                                                      │
│  seed:                                                              │
│  ────                                                               │
│  DataLoader 随机种子                                                │
│                                                                      │
│  更新过程：                                                          │
│  ─────────                                                          │
│                                                                      │
│  for epoch in range(ppo_epochs):                                   │
│      for mini_batch in mini_batches:                               │
│          loss = compute_ppo_loss(mini_batch)                       │
│          optimizer.step()                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、_update_critic 方法

### 6.1 方法定义

```python
def _update_critic(self, batch: DataProto) -> DataProto:
    """Update Critic value function using WorkerGroup."""
```

**核心职责：**
- 调用 Critic WorkerGroup 执行价值函数更新

### 6.2 代码详解

```python
def _update_critic(self, batch: DataProto) -> DataProto:
    if self.use_legacy_worker_impl == "disable":
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_epochs = self.config.critic.ppo_epochs
        
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            ...
        )
        
        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
    else:
        critic_output = self.critic_wg.update_critic(batch)
    return critic_output
```

### 6.3 Critic Loss

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Critic Loss 计算                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Value Loss:                                                        │
│  ─────────                                                          │
│  L_value = MSE(V(s), R)                                            │
│                                                                      │
│  其中：                                                              │
│  - V(s) = Critic 预测的价值                                        │
│  - R = Advantage 计算的 return                                     │
│                                                                      │
│  目标：                                                              │
│  ─────                                                              │
│  让 Critic 学会准确预测状态价值                                    │
│                                                                      │
│  更新过程：                                                          │
│  ─────────                                                          │
│                                                                      │
│  for epoch in range(ppo_epochs):                                   │
│      for mini_batch in mini_batches:                               │
│          values = critic(mini_batch)                               │
│          returns = mini_batch["returns"]                           │
│          loss = MSE(values, returns)                               │
│          optimizer.step()                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、新旧版本对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                  新旧版本计算方法对比                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  旧版 Worker (use_legacy_worker_impl = "auto/enable"):             │
│  ─────────────────────────────────────────────                      │
│                                                                      │
│  - 直接使用 DataProto                                              │
│  - Worker 内部处理 padding                                         │
│  - 专用方法名：compute_values, update_critic                       │
│                                                                      │
│  新版引擎 (use_legacy_worker_impl = "disable"):                     │
│  ───────────────────────────────────────                            │
│                                                                      │
│  - 需要转换为 TensorDict                                           │
│  - 需要 padding → nopadding 转换                                   │
│  - 通用方法名：infer_batch, train_mini_batch                       │
│                                                                      │
│  转换流程：                                                          │
│  ─────────                                                          │
│                                                                      │
│  DataProto → TensorDict → NoPadding → RPC → NoPadding → Padding → │
│  TensorDict → DataProto                                             │
│                                                                      │
│  源码注释：                                                          │
│  ─────────                                                          │
│  "TODO: remove step 1, 2, 4 after we make the whole training       │
│   tensordict and padding free"                                      │
│                                                                      │
│  翻译：                                                              │
│  "TODO: 在整个训练 tensordict 和 padding free 后，                 │
│   移除 step 1, 2, 4"                                                │
│                                                                      │
│  说明：                                                              │
│  目前转换是临时方案                                                  │
│  未来会统一为 tensordict + padding free                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 八、总结

| 方法 | 新版引擎 | 旧版 Worker |
|------|----------|-------------|
| `_compute_values` | `infer_batch` + TensorDict | `compute_values` |
| `_compute_ref_log_prob` | `compute_ref_log_prob` + TensorDict | `compute_ref_log_prob` |
| `_compute_old_log_prob` | `compute_log_prob` + TensorDict | `compute_log_prob` |
| `_update_actor` | `update_actor` + TensorDict | `update_actor` |
| `_update_critic` | `train_mini_batch` + TensorDict | `update_critic` |

**一句话总结：**
五个计算方法封装了新版引擎和旧版 Worker 的不同调用方式，统一接口供 fit() 调用，新版需要额外的 TensorDict 和 Padding 转换。