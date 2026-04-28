# RayDAPOTrainer.fit() 详细流程解析

> verl/recipe/dapo/dapo_ray_trainer.py

## 一、整体流程概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RayDAPOTrainer.fit() 流程                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  初始化阶段                                                          │
│  ─────────                                                          │
│  1. 创建 Logger (Tracking)                                          │
│  2. global_steps = 0, gen_steps = 0                                 │
│  3. _load_checkpoint()                                              │
│  4. _validate() (可选，val_before_train)                            │
│  5. 初始化 RolloutSkip (可选)                                       │
│                                                                      │
│  主循环                                                              │
│  ───────                                                            │
│  for epoch in range(total_epochs):                                  │
│      for batch_dict in train_dataloader:                            │
│          │                                                           │
│          │  ┌─────────────────────────────────────────────────────┐ │
│          │  │           DAPO 动态采样循环 (核心)                    │ │
│          │  │                                                     │ │
│          │  │  while num_prompt_in_batch < train_batch_size:     │ │
│          │  │      Step 1: generate_sequences()                  │ │
│          │  │      Step 2: union batch                           │ │
│          │  │      Step 3: compute_reward()                      │ │
│          │  │      Step 4: filter_groups (动态采样核心)           │ │
│          │  │      Step 5: 检查样本数量                           │ │
│          │  │          - 不够: continue (重新采样)               │ │
│          │  │          - 够了: break (开始训练)                  │ │
│          │  │                                                     │ │
│          │  └─────────────────────────────────────────────────────┘ │
│          │                                                           │
│          │  标准 PPO 训练步骤                                        │
│          │  ───────────────────                                    │
│          │  Step 6: compute_kl_related_metrics                     │
│          │  Step 7: compute_values (可选)                          │
│          │  Step 8: compute_advantage                              │
│          │  Step 9: update_critic (可选)                           │
│          │  Step 10: update_actor                                  │
│          │                                                           │
│          │  后处理                                                   │
│          │  ────────                                                │
│          │  - 验证 (test_freq)                                     │
│          │  - 保存 checkpoint (save_freq)                          │
│          │  - 记录 metrics                                         │
│          │  - 重置 batch, num_prompt_in_batch, num_gen_batches    │
│          │                                                           │
│          global_steps += 1                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、初始化阶段详解

### 2.1 源码位置

```python
# 第 83-133 行
```

### 2.2 初始化代码

```python
from verl.utils.tracking import Tracking
from omegaconf import OmegaConf

# 1. 创建 Logger
logger = Tracking(
    project_name=self.config.trainer.project_name,
    experiment_name=self.config.trainer.experiment_name,
    default_backend=self.config.trainer.logger,  # wandb/swanlab/etc
    config=OmegaConf.to_container(self.config, resolve=True),
)

# 2. 初始化计数器
self.global_steps = 0   # 训练步数
self.gen_steps = 0      # 生成步数（包括被过滤的）

# 3. 加载 checkpoint
self._load_checkpoint()

# 4. 初始验证（可选）
if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
    val_metrics = self._validate()
    logger.log(data=val_metrics, step=self.global_steps)
    if self.config.trainer.get("val_only", False):
        return  # 只验证，不训练

# 5. 初始化进度条
progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps)

# 6. 初始化动态采样相关变量
batch = None              # 累积的有效样本
num_prompt_in_batch = 0   # 当前累积的有效 prompt 数
num_gen_batches = 0       # 当前已生成的 batch 数
```

### 2.3 计数器含义

| 计数器 | 含义 | 更新时机 |
|--------|------|----------|
| `global_steps` | 实际训练步数 | 完成一次 actor+critic 更新后 +1 |
| `gen_steps` | 生成步数 | 每次调用 generate_sequences 后 +1 |
| `num_gen_batches` | 当前 step 的生成次数 | 每次生成后 +1，完成训练后重置为 0 |

---

## 三、动态采样循环详解（DAPO 核心）

### 3.1 Step 1: 生成序列

```python
# 第 147-161 行

# 从 dataloader 获取一个 batch
new_batch: DataProto = DataProto.from_single_dict(batch_dict)
num_gen_batches += 1

# 准备生成用的 batch
gen_batch = self._get_gen_batch(new_batch)

# 每个 prompt 生成 n 个 response
gen_batch_output = gen_batch.repeat(
    repeat_times=self.config.actor_rollout_ref.rollout.n,  # 如 n=4
    interleave=True
)

# 调用 rollout 生成
with marked_timer("gen", timing_raw, "red"):
    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
```

**repeat 的作用**：

```
原始: [prompt_1, prompt_2, prompt_3]  (3个prompt)
repeat(n=4, interleave=True):
结果: [prompt_1, prompt_1, prompt_1, prompt_1,
       prompt_2, prompt_2, prompt_2, prompt_2,
       prompt_3, prompt_3, prompt_3, prompt_3]  (12个)

每个 prompt 会生成 4 个 response
```

---

### 3.2 Step 2: 合并 batch

```python
# 第 187-192 行

# 添加唯一 ID
new_batch.non_tensor_batch["uid"] = np.array(
    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
)

# repeat 原始 batch 以对齐生成的 response 数量
new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

# 合并生成结果
new_batch = new_batch.union(gen_batch_output)
```

**数据结构变化**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Batch 数据结构变化                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  new_batch (原始):                                                   │
│  ─────────────────                                                  │
│  batch:                                                              │
│    prompts: [p1, p2, p3]                                            │
│  non_tensor_batch:                                                   │
│    uid: [uuid_1, uuid_2, uuid_3]                                    │
│                                                                      │
│  new_batch.repeat(n=4) 后:                                          │
│  ──────────────────────────                                         │
│  batch:                                                              │
│    prompts: [p1, p1, p1, p1, p2, p2, p2, p2, p3, p3, p3, p3]        │
│  non_tensor_batch:                                                   │
│    uid: [uuid_1, uuid_1, uuid_1, uuid_1, uuid_2, ...]              │
│                                                                      │
│  gen_batch_output (生成结果):                                        │
│  ─────────────────────────────                                      │
│  batch:                                                              │
│    responses: [r1_1, r1_2, r1_3, r1_4, r2_1, ...]                   │
│    input_ids: [...]                                                  │
│    attention_mask: [...]                                             │
│                                                                      │
│  new_batch.union(gen_batch_output) 后:                              │
│  ─────────────────────────────────────                              │
│  batch:                                                              │
│    prompts: [p1, p1, p1, p1, ...]                                   │
│    responses: [r1_1, r1_2, r1_3, r1_4, ...]                         │
│    input_ids: [...]                                                  │
│    attention_mask: [...]                                             │
│  non_tensor_batch:                                                   │
│    uid: [uuid_1, uuid_1, uuid_1, uuid_1, ...]                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.3 Step 3: 计算 Reward

```python
# 第 199-227 行

with marked_timer("reward", timing_raw, "yellow"):
    # 如果使用 Reward Model
    if self.use_rm and "rm_scores" not in new_batch.batch.keys():
        reward_tensor = self.rm_wg.compute_rm_score(new_batch)
        new_batch = new_batch.union(reward_tensor)

    # 计算 rule-based reward 或组合 reward
    reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

    # 存储 token level scores
    new_batch.batch["token_level_scores"] = reward_tensor

    # 如果有额外信息（如 accuracy）
    if reward_extra_infos_dict:
        new_batch.non_tensor_batch.update(
            {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
        )

    # KL penalty (可选)
    if self.config.algorithm.use_kl_in_reward:
        new_batch, kl_metrics = apply_kl_penalty(new_batch, ...)
        metrics.update(kl_metrics)
    else:
        new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
```

---

### 3.4 Step 4: 动态采样（核心）

```python
# 第 229-288 行

if not self.config.algorithm.filter_groups.enable:
    # 不启用动态采样：直接使用
    batch = new_batch
else:
    # 启用动态采样：DAPO 核心
    
    # 1. 确定过滤指标
    metric_name = self.config.algorithm.filter_groups.metric  # "acc" 或 "seq_reward"
    
    # 2. 计算每个序列的 reward 总和
    if metric_name == "seq_final_reward":
        new_batch.non_tensor_batch["seq_final_reward"] = (
            new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
        )
    elif metric_name == "seq_reward":
        new_batch.non_tensor_batch["seq_reward"] = (
            new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
        )

    # 3. 按 prompt_uid 收集每个 prompt 的所有 response 的 metric
    prompt_uid2metric_vals = defaultdict(list)
    for uid, metric_val in zip(
        new_batch.non_tensor_batch["uid"], 
        new_batch.non_tensor_batch[metric_name]
    ):
        prompt_uid2metric_vals[uid].append(metric_val)
    
    # 4. 计算每个 prompt 的 metric 的 std
    prompt_uid2metric_std = {}
    for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
        prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

    # 5. 过滤：只保留 std > 0 的 prompt（有差异的）
    #    或者只有 1 个 response 的 prompt
    kept_prompt_uids = [
        uid for uid, std in prompt_uid2metric_std.items()
        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
    ]
    num_prompt_in_batch += len(kept_prompt_uids)

    # 6. 保留有效样本
    kept_traj_idxs = []
    for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
        if traj_from_prompt_uid in kept_prompt_uids:
            kept_traj_idxs.append(idx)
    
    new_batch = new_batch[kept_traj_idxs]
    
    # 7. 累积到 batch
    batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

    # 8. 检查样本数量
    prompt_bsz = self.config.data.train_batch_size
    if num_prompt_in_batch < prompt_bsz:
        # 样本不够，继续采样
        print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
        max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
        
        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
            # 继续生成
            self.gen_steps += 1
            continue  # ← 关键：跳回循环开头，取下一个 batch_dict
        else:
            raise ValueError("Generated too many. Data too difficult?")
    else:
        # 样本够了，对齐 batch size
        traj_bsz = train_batch_size * n
        batch = batch[:traj_bsz]
```

---

### 3.5 动态采样流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    动态采样详细流程                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入: train_batch_size = 16, n = 4                                 │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  第一次生成 (num_gen_batches = 1)                              │ │
│  │                                                               │ │
│  │  prompts: [p1, p2, p3, p4]                                    │ │
│  │                                                               │ │
│  │  生成 4×4 = 16 个 response                                    │ │
│  │                                                               │ │
│  │  计算 reward:                                                 │ │
│  │  p1: [r=1, r=1, r=1, r=1] → std=0 → 过滤                    │ │
│  │  p2: [r=0, r=0, r=0, r=0] → std=0 → 过滤                    │ │
│  │  p3: [r=1, r=0, r=1, r=0] → std=0.5 → 保留                  │ │
│  │  p4: [r=0.8, r=0.6, r=0.9, r=0.7] → std=0.12 → 保留        │ │
│  │                                                               │ │
│  │  有效 prompt: p3, p4 → num_prompt_in_batch = 2               │ │
│  │                                                               │ │
│  │  需要 16 个，只有 2 个 → 不够                                  │ │
│  │                                                               │ │
│  │  continue → 取下一个 batch_dict                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  第二次生成 (num_gen_batches = 2)                              │ │
│  │                                                               │ │
│  │  prompts: [p5, p6, p7, p8]                                    │ │
│  │                                                               │ │
│  │  计算 reward:                                                 │ │
│  │  p5: [r=1, r=0, r=0.5, r=1] → std > 0 → 保留                │ │
│  │  p6: [r=0, r=1, r=0, r=1] → std > 0 → 保留                  │ │
│  │  p7: [r=1, r=1, r=1, r=1] → std=0 → 过滤                    │ │
│  │  p8: [r=0.5, r=0.5, r=0.5, r=0.5] → std=0 → 过滤           │ │
│  │                                                               │ │
│  │  有效 prompt: p5, p6 → num_prompt_in_batch = 2 + 2 = 4       │ │
│  │                                                               │ │
│  │  batch 现有: [p3 的 4 个 response, p4 的 4 个 response,      │ │
│  │              p5 的 4 个 response, p6 的 4 个 response]       │ │
│  │                                                               │ │
│  │  需要 16 个，只有 4 个 → 不够                                  │ │
│  │                                                               │ │
│  │  continue → 取下一个 batch_dict                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ... 继续 ...                                                        │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  第八次生成 (num_gen_batches = 8)                              │ │
│  │                                                               │ │
│  │  num_prompt_in_batch = 18 ≥ 16 → 够了！                       │ │
│  │                                                               │ │
│  │  batch[:16*4] → 截取前 64 个 trajectory                       │ │
│  │                                                               │ │
│  │  开始标准 PPO 训练                                             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.6 过滤逻辑详解

```
┌─────────────────────────────────────────────────────────────────────┐
│                    过滤逻辑详解                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  原理：                                                              │
│  ─────                                                              │
│                                                                      │
│  每个 prompt 生成 n 个 response，每个 response 有一个 reward        │
│                                                                      │
│  计算这 n 个 reward 的标准差 (std):                                  │
│                                                                      │
│  std = 0:                                                           │
│  ───────                                                            │
│  所有 response 的 reward 相同                                        │
│  → 全对 (reward = 1) 或 全错 (reward = 0)                           │
│  → 无法区分哪个 response 更好                                       │
│  → Advantage 无法计算                                               │
│  → 过滤掉                                                           │
│                                                                      │
│  std > 0:                                                           │
│  ───────                                                            │
│  response 之间有差异                                                 │
│  → 有些对有些错，或者准确率不同                                      │
│  → 可以学习"什么样的回答更好"                                       │
│  → 保留                                                             │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  特殊情况：                                                          │
│  ─────────                                                          │
│                                                                      │
│  len(prompt_uid2metric_vals[uid]) == 1:                             │
│  → 只有 1 个 response                                               │
│  → 无法计算 std (数学上 std = 0，但单样本应该保留)                  │
│  → 保留                                                             │
│                                                                      │
│  这通常发生在 n=1 的情况                                             │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  为什么需要动态采样：                                                 │
│  ─────────────────────                                              │
│                                                                      │
│  GRPO 的 Advantage 计算：                                            │
│  A_i = (r_i - mean(r_group)) / std(r_group)                         │
│                                                                      │
│  如果 std(r_group) = 0:                                             │
│  → A_i = 0 / 0 = undefined                                         │
│  → 无法计算 Advantage                                               │
│  → 无法进行策略更新                                                  │
│                                                                      │
│  直觉理解：                                                          │
│  ─────                                                              │
│  "没有对比就没有学习"                                                │
│  - 全对：模型已经掌握，不需要再学                                    │
│  - 全错：模型完全不会，当前的 n 个 response 都不行                   │
│  - 有差异：模型可以从差异中学习                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、标准 PPO 训练步骤

```python
# 第 290-346 行

# 4.1 平衡 batch（可选）
if self.config.trainer.balance_batch:
    self._balance_batch(batch, metrics=metrics)

# 4.2 计算 KL 相关 metrics
if not self.config.algorithm.use_kl_in_reward:
    batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

# 4.3 计算 Values（如果使用 Critic）
if self.use_critic:
    with marked_timer("values", timing_raw, "cyan"):
        values = self.critic_wg.compute_values(batch)
        batch = batch.union(values)

# 4.4 计算 Advantage
with marked_timer("adv", timing_raw, "brown"):
    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,  # GAE/GRPO
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
    )

# 4.5 更新 Critic（可选）
if self.use_critic:
    with marked_timer("update_critic", timing_raw, "pink"):
        critic_output = self.critic_wg.update_critic(batch)
    metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

# 4.6 更新 Actor（检查 critic_warmup）
if self.config.trainer.critic_warmup <= self.global_steps:
    with marked_timer("update_actor", timing_raw, "red"):
        actor_output = self.actor_rollout_wg.update_actor(batch)
    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
```

---

## 五、验证和保存

```python
# 第 352-396 行

# 5.1 验证
if self.val_reward_fn is not None and \
   self.config.trainer.test_freq > 0 and \
   (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
    with marked_timer("testing", timing_raw, "green"):
        val_metrics = self._validate()
    metrics.update(val_metrics)

# 5.2 保存 checkpoint
if self.config.trainer.save_freq > 0 and \
   (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
    with marked_timer("save_checkpoint", timing_raw, "green"):
        self._save_checkpoint()

# 5.3 记录 metrics
metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

# 5.4 记录动态采样相关 metric
metrics["train/num_gen_batches"] = num_gen_batches

# 5.5 重置动态采样状态
batch = None
num_prompt_in_batch = 0
num_gen_batches = 0

# 5.6 记录到 logger
logger.log(data=metrics, step=self.global_steps)

# 5.7 更新进度
progress_bar.update(1)
self.global_steps += 1
self.gen_steps += 1
```

---

## 六、与标准 RayPPOTrainer.fit() 的对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DAPO vs PPO fit() 对比                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  RayPPOTrainer.fit():                                               │
│  ─────────────────────                                              │
│                                                                      │
│  for batch_dict in dataloader:                                      │
│      generate_sequences()                                           │
│      compute_reward()                                               │
│      compute_advantage()                                            │
│      update_actor()                                                 │
│      update_critic()                                                │
│      global_steps += 1                                              │
│                                                                      │
│  每个 batch_dict 对应 1 个 training step                            │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  RayDAPOTrainer.fit():                                              │
│  ─────────────────────                                              │
│                                                                      │
│  for batch_dict in dataloader:                                      │
│      generate_sequences()                                           │
│      compute_reward()                                               │
│      filter_groups():                                               │
│          if std(rewards) == 0:                                      │
│              continue  # 跳过，取下一个 batch_dict                  │
│          else:                                                      │
│              accumulate batch                                       │
│      if num_prompts < train_batch_size:                             │
│          continue  # 不够，继续采样                                  │
│      else:                                                          │
│          compute_advantage()                                        │
│          update_actor()                                             │
│          update_critic()                                            │
│          global_steps += 1                                          │
│                                                                      │
│  多个 batch_dict 可能对应 1 个 training step                        │
│  num_gen_batches 记录实际生成了多少次                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、关键配置项

```yaml
algorithm:
  filter_groups:
    enable: true              # 启用动态采样
    metric: acc               # 过滤指标（acc/seq_reward/seq_final_reward）
    max_num_gen_batches: 10   # 最大生成次数（0 = 无限制）

  adv_estimator: grpo         # Advantage 计算方式
  use_kl_in_reward: false     # KL 是否加入 reward

actor_rollout_ref:
  rollout:
    n: 4                      # 每个 prompt 生成 4 个 response

data:
  train_batch_size: 16        # 需要的有效 prompt 数量
```

### 7.1 filter_groups.metric 选项

| metric | 含义 |
|--------|------|
| `acc` | 使用 reward_extra_infos_dict 中的 accuracy 字段 |
| `seq_reward` | token_level_scores 的总和 |
| `seq_final_reward` | token_level_rewards 的总和（包含 KL penalty） |

### 7.2 max_num_gen_batches 含义

| 值 | 行为 |
|----|------|
| `0` | 无限制，一直生成直到满足 train_batch_size |
| `N > 0` | 最多生成 N 次，超过则报错 |

---

## 八、compute_kl_related_metrics 方法

```python
# 第 50-74 行

def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
    # 1. 计算 response_mask
    batch.batch["response_mask"] = compute_response_mask(batch)

    # 2. 计算 old_log_probs
    with marked_timer("old_log_prob", timing_raw, "blue"):
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
        
        # 计算 entropy
        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
        metrics.update(old_log_prob_metrics)
        
        old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)

    # 3. 计算 ref_log_prob（可选）
    if self.use_reference_policy:
        with marked_timer("ref", timing_raw, "olive"):
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

    return batch
```

---

## 九、总结

| 特性 | RayPPOTrainer | RayDAPOTrainer |
|------|---------------|----------------|
| 训练循环 | 简单 for 循环 | 动态采样循环 |
| batch_dict 对应 | 1 training step | 可能多个 batch_dict = 1 step |
| 过滤逻辑 | 无 | std(rewards) == 0 则过滤 |
| 样本累积 | 无 | 累积有效样本直到满足 train_batch_size |
| 特殊 metric | 无 | train/num_gen_batches |

**一句话总结：**

RayDAPOTrainer.fit() 的核心是动态采样循环：每次生成后检查每个 prompt 的多个 response 的 reward std，过滤掉全对/全错的 prompt，累积有效样本直到达到 train_batch_size，然后执行标准 PPO 训练步骤。这使得多个 dataloader batch 可能对应一个 training step。