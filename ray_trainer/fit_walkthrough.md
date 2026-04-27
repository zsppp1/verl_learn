# fit() 方法详解

> verl/trainer/ppo/ray_trainer.py 行1272-1655

## 一、方法定义

```python
def fit(self):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC
    to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """
```

**核心职责：**
- 执行完整的 PPO 训练循环
- 通过 RPC 调用 WorkerGroup 的计算函数
- 在 Driver 进程执行 Advantage 计算

---

## 二、训练循环整体结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    fit() 训练循环结构                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  初始化阶段：                                                        │
│  ─────────                                                          │
│  1. 创建 logger (Tracking)                                         │
│  2. 加载 checkpoint (_load_checkpoint)                             │
│  3. 初始验证 (_validate, 可选)                                     │
│                                                                      │
│  主循环：                                                            │
│  ───────                                                            │
│  for epoch in range(total_epochs):                                 │
│      for batch in train_dataloader:                                │
│          │                                                           │
│          │  ┌─────────────────────────────────────────────────────┐ │
│          │  │              PPO 训练步骤                             │ │
│          │  │                                                     │ │
│          │  │  1. generate_sequences (Rollout)                   │ │
│          │  │  2. compute_reward                                  │ │
│          │  │  3. compute_old_log_prob                            │ │
│          │  │  4. compute_ref_log_prob (可选)                     │ │
│          │  │  5. compute_values (可选)                           │ │
│          │  │  6. apply_kl_penalty (可选)                         │ │
│          │  │  7. compute_advantage                               │ │
│          │  │  8. update_critic (可选)                            │ │
│          │  │  9. update_actor                                    │ │
│          │  │                                                     │ │
│          │  └─────────────────────────────────────────────────────┘ │
│          │                                                           │
│          │  后处理：                                                 │
│          │  ────────                                                │
│          │  - 验证 (test_freq)                                     │
│          │  - 保存 checkpoint (save_freq)                          │
│          │  - 记录 metrics                                         │
│          │                                                           │
│          global_steps += 1                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、详细代码解析

### 3.1 初始化阶段

```python
from verl.utils.tracking import Tracking

logger = Tracking(
    project_name=self.config.trainer.project_name,
    experiment_name=self.config.trainer.experiment_name,
    default_backend=self.config.trainer.logger,
    config=OmegaConf.to_container(self.config, resolve=True),
)

self.global_steps = 0
self._load_checkpoint()

current_epoch = self.global_steps // len(self.train_dataloader)
```

**global_steps 管理：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  global_steps vs epoch                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  global_steps = 已训练的 batch 数量                                 │
│  epoch = global_steps // len(train_dataloader)                      │
│                                                                      │
│  示例：                                                              │
│  ─────                                                              │
│  train_dataloader 有 100 batch                                     │
│                                                                      │
│  global_steps=0   → epoch=0                                        │
│  global_steps=50  → epoch=0                                        │
│  global_steps=100 → epoch=1                                        │
│  global_steps=200 → epoch=2                                        │
│                                                                      │
│  计算方式：                                                          │
│  ─────────                                                          │
│  total_training_steps = len(train_dataloader) × total_epochs       │
│                                                                      │
│  例如：                                                              │
│  100 batch × 10 epochs = 1000 steps                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 初始验证

```python
if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
    val_metrics = self._validate()
    assert val_metrics, f"{val_metrics=}"
    pprint(f"Initial validation metrics: {val_metrics}")
    logger.log(data=val_metrics, step=self.global_steps)
    if self.config.trainer.get("val_only", False):
        return
```

**val_only 模式：**

```
仅验证模式，不训练
用于快速测试验证流程
```

---

### 3.3 主循环结构

```python
progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

self.global_steps += 1  # we start from step 1

for epoch in range(current_epoch, self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        metrics = {}
        timing_raw = {}
        
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        
        # ... PPO 训练步骤
        
        logger.log(data=metrics, step=self.global_steps)
        progress_bar.update(1)
        self.global_steps += 1
```

---

### 3.4 PPO 训练步骤详解

#### Step 1: Generate Rollout

```python
batch: DataProto = DataProto.from_single_dict(batch_dict)
batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

# add uid to batch
batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

gen_batch = self._get_gen_batch(batch)
gen_batch.meta_info["global_steps"] = self.global_steps

gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

with marked_timer("gen", timing_raw, color="red"):
    if not self.async_rollout_mode:
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
    else:
        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
```

**rollout.n 参数：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  rollout.n (采样数)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  n = 每个 prompt 生成多少个 response                                │
│                                                                      │
│  示例：                                                              │
│  ─────                                                              │
│  batch 有 10 个 prompt                                              │
│  n = 4                                                              │
│                                                                      │
│  gen_batch.repeat(n=4, interleave=True)                            │
│                                                                      │
│  结果：                                                              │
│  [prompt_1, prompt_1, prompt_1, prompt_1,                          │
│   prompt_2, prompt_2, prompt_2, prompt_2, ...]                     │
│                                                                      │
│  共 40 个 prompt                                                    │
│  每个 prompt 生成 4 个 response                                    │
│                                                                      │
│  用途：                                                              │
│  ─────                                                              │
│  - GRPO 需要多个 response 比较                                     │
│  - 提高样本多样性                                                   │
│  - 更稳定的训练                                                    │
│                                                                      │
│  interleave=True:                                                   │
│  ─────────────                                                      │
│  prompt 按顺序重复：[p1, p1, p1, p1, p2, p2, p2, p2, ...]          │
│                                                                      │
│  interleave=False:                                                  │
│  ─────────────                                                      │
│  prompt 按批次重复：[p1, p2, p3, p1, p2, p3, ...]                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### Step 2: Union Rollout Output

```python
batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
batch = batch.union(gen_batch_output)
```

**DataProto.union 作用：**

```
将 gen_batch_output 的数据合并到 batch 中

batch 原本只有 prompt
gen_batch_output 有 prompt + response

union 后:
batch = {
    "prompts": [...],
    "responses": [...],       # 新增
    "attention_mask": [...],  # 新增
    ...
}
```

---

#### Step 3: Compute Reward

```python
with marked_timer("reward", timing_raw, color="yellow"):
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        if not self.use_reward_loop:
            reward_tensor = self.rm_wg.compute_rm_score(batch)
        else:
            reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
        batch = batch.union(reward_tensor)
    
    if self.config.reward_model.launch_reward_fn_async:
        future_reward = compute_reward_async.remote(data=batch, ...)
    else:
        reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(batch, reward_fn=self.reward_fn)
```

**两种 reward 计算方式：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Reward 计算方式                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Reward Model (self.use_rm):                                     │
│  ─────────────────────────────                                      │
│  - 使用 rm_wg.compute_rm_score(batch)                              │
│  - 模型推理计算 reward                                             │
│                                                                      │
│  2. Reward Function (reward_fn):                                    │
│  ─────────────────────────────                                      │
│  - 使用 _compute_or_extract_reward(batch, reward_fn)               │
│  - 规则奖励或其他计算方式                                          │
│                                                                      │
│  异步模式 (launch_reward_fn_async):                                 │
│  ─────────────────────────────                                      │
│  - compute_reward_async.remote()                                   │
│  - reward 计算与后续步骤并行                                       │
│  - 后续 ray.get(future_reward) 获取结果                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### Step 4: Compute old_log_prob

```python
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
    entropys = old_log_prob.batch["entropys"]
    response_masks = batch.batch["response_mask"]
    
    entropy_agg = agg_loss(
        loss_mat=entropys,
        loss_mask=response_masks,
        loss_agg_mode=actor_config.loss_agg_mode,
    )
    
    old_log_prob_metrics = {
        "actor/entropy": entropy_agg.detach().item(),
        "perf/mfu/actor_infer": old_log_prob_mfu,
    }
    metrics.update(old_log_prob_metrics)
    
    batch = batch.union(old_log_prob)
```

**old_log_prob 的作用：**

```
PPO 的 importance sampling 需要：

ratio = π_new(a|s) / π_old(a|s)

old_log_prob = log π_old(a|s)
保存当前策略的 log_prob，作为后续更新的锚点
```

---

#### Step 5: Compute ref_log_prob (可选)

```python
if self.use_reference_policy:
    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
        ref_log_prob = self._compute_ref_log_prob(batch)
        batch = batch.union(ref_log_prob)
```

---

#### Step 6: Compute Values (可选)

```python
if self.use_critic:
    with marked_timer("values", timing_raw, color="cyan"):
        values = self._compute_values(batch)
        batch = batch.union(values)
```

---

#### Step 7: Apply KL Penalty + Compute Advantage

```python
with marked_timer("adv", timing_raw, color="brown"):
    batch.batch["token_level_scores"] = reward_tensor
    
    if self.config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward)
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
    
    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
    )
```

---

#### Step 8: Update Critic (可选)

```python
if self.use_critic:
    with marked_timer("update_critic", timing_raw, color="pink"):
        critic_output = self._update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)
```

---

#### Step 9: Update Actor

```python
if self.config.trainer.critic_warmup <= self.global_steps:
    with marked_timer("update_actor", timing_raw, color="red"):
        actor_output = self._update_actor(batch)
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)
```

**critic_warmup：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  critic_warmup 预热                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  critic_warmup = N                                                  │
│                                                                      │
│  前 N steps:                                                        │
│  ─────────                                                          │
│  - 只更新 Critic                                                    │
│  - 不更新 Actor                                                    │
│                                                                      │
│  原因：                                                              │
│  ─────                                                              │
│  - Critic 需要先学会准确估计 Value                                 │
│  - 否则 Advantage 计算不稳定                                       │
│  - Actor 更新会出问题                                              │
│                                                                      │
│  N steps 后:                                                        │
│  ─────────                                                          │
│  - Actor 和 Critic 都更新                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.5 验证和保存

```python
# validate
if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
   (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
    with marked_timer("testing", timing_raw, color="green"):
        val_metrics = self._validate()
    metrics.update(val_metrics)

# save checkpoint
if self.config.trainer.save_freq > 0 and \
   (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
    with marked_timer("save_checkpoint", timing_raw, color="green"):
        self._save_checkpoint()
```

---

### 3.6 Metrics 记录

```python
metrics.update({
    "training/global_step": self.global_steps,
    "training/epoch": epoch,
})
metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

logger.log(data=metrics, step=self.global_steps)
```

---

## 四、时序图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    fit() 时序图                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Driver Process                                                      │
│  ────────────                                                        │
│                                                                      │
│  batch_dict ──→ DataProto ──→ gen_batch                            │
│                                          │                           │
│                                          │ repeat(n)                 │
│                                          ↓                           │
│                                          gen_batch_output            │
│                                          │                           │
│                                          ↓                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     actor_rollout_wg.generate_sequences         ││
│  │                                                                 ││
│  │   Worker 1 (GPU 0): generate part of batch                     ││
│  │   Worker 2 (GPU 1): generate part of batch                     ││
│  │   Worker 3 (GPU 2): generate part of batch                     ││
│  │   Worker 4 (GPU 3): generate part of batch                     ││
│  │                                                                 ││
│  │   汇总返回 gen_batch_output                                    ││
│  │                                                                 ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                          │                           │
│                                          ↓                           │
│  batch.union(gen_batch_output)                                      │
│                                          │                           │
│                                          ↓                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     _compute_or_extract_reward                  ││
│  │                                                                 ││
│  │   reward_fn(batch) → reward_tensor                             ││
│  │                                                                 ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                          │                           │
│                                          ↓                           │
│  batch["token_level_scores"] = reward_tensor                        │
│                                          │                           │
│                                          ↓                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     _compute_old_log_prob                       ││
│  │                                                                 ││
│  │   actor_rollout_wg.compute_log_prob(batch)                     ││
│  │                                                                 ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                          │                           │
│                                          ↓                           │
│  batch.union(old_log_prob)                                          │
│                                          │                           │
│                                          ↓                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     _compute_values (if use_critic)            ││
│  │                                                                 ││
│  │   critic_wg.compute_values(batch)                              ││
│  │                                                                 ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                          │                           │
│                                          ↓                           │
│  batch.union(values)                                                │
│                                          │                           │
│                                          ↓                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     compute_advantage (Driver本地)              ││
│  │                                                                 ││
│  │   GAE/GRPO 计算，不需要 RPC                                    ││
│  │                                                                 ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                          │                           │
│                                          ↓                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     _update_critic                              ││
│  │                                                                 ││
│  │   critic_wg.update_critic(batch)                               ││
│  │                                                                 ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                          │                           │
│                                          ↓                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     _update_actor                               ││
│  │                                                                 ││
│  │   actor_rollout_wg.update_actor(batch)                         ││
│  │                                                                 ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                          │                           │
│                                          ↓                           │
│  logger.log(metrics)                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 五、总结

| 组件 | 执行位置 |
|------|----------|
| generate_sequences | WorkerGroup (RPC) |
| compute_reward | WorkerGroup 或 Driver |
| compute_log_prob | WorkerGroup (RPC) |
| compute_values | WorkerGroup (RPC) |
| compute_advantage | Driver (本地) |
| update_actor | WorkerGroup (RPC) |
| update_critic | WorkerGroup (RPC) |

**一句话总结：**
`fit()` 是 PPO 训练的核心循环，通过 RPC 调用 WorkerGroup 执行分布式计算，在 Driver 进程执行轻量级 Advantage 计算，实现高效的分布式 PPO 训练。