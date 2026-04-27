# Checkpoint 和验证方法详解

> verl/trainer/ppo/ray_trainer.py 行938-1063, 607-749

## 一、Checkpoint 方法

### 1.1 _save_checkpoint 方法

```python
def _save_checkpoint(self):
    """Save Actor, Critic, and DataLoader state to checkpoint."""
```

**核心职责：**
- 保存 Actor 和 Critic 模型参数
- 保存 DataLoader 状态
- 记录 global_steps

### 1.2 代码详解

```python
def _save_checkpoint(self):
    from verl.utils.fs import local_mkdir_safe
    
    # path: given_path + `/global_step_{global_steps}` + `/actor`
    local_global_step_folder = os.path.join(
        self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
    )
    
    actor_local_path = os.path.join(local_global_step_folder, "actor")
    actor_remote_path = (
        None
        if self.config.trainer.default_hdfs_dir is None
        else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
    )
    
    max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
    
    self.actor_rollout_wg.save_checkpoint(
        actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
    )
    
    if self.use_critic:
        critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
        self.critic_wg.save_checkpoint(
            critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
        )
    
    # save dataloader
    local_mkdir_safe(local_global_step_folder)
    dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
    dataloader_state_dict = self.train_dataloader.state_dict()
    torch.save(dataloader_state_dict, dataloader_local_path)
    
    # latest checkpointed iteration tracker
    local_latest_checkpointed_iteration = os.path.join(
        self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
    )
    with open(local_latest_checkpointed_iteration, "w") as f:
        f.write(str(self.global_steps))
```

### 1.3 Checkpoint 文件结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Checkpoint 文件结构                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  checkpoint_dir/                                                    │
│  ├── global_step_100/                                               │
│  │   ├── actor/                # Actor 模型                        │
│  │   │   ├── model.pt                                             │
│  │   │   ├── optimizer.pt                                         │
│  │   │   └────────── ...                                          │
│  │   ├── Critic/               # Critic 模型                       │
│  │   │   ├── model.pt                                             │
│  │   │   ├── optimizer.pt                                         │
│  │   ├── data.pt               # DataLoader 状态                  │
│  │                                                                  │
│  ├── global_step_200/                                               │
│  │   ├── ...                                                        │
│  │                                                                  │
│  ├── latest_checkpointed_iteration.txt  # 最新 checkpoint 记录     │
│  │   内容: "200"                                                    │
│  │                                                                  │
│  max_actor_ckpt_to_keep:                                            │
│  ───────────────────                                                │
│  - None: 保存所有 checkpoint                                       │
│  - N: 只保留最近 N 个 checkpoint                                   │
│                                                                      │
│  作用：                                                              │
│  ─────                                                              │
│  - 避免磁盘空间无限增长                                              │
│  - 自动删除旧的 checkpoint                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1.4 _load_checkpoint 方法

```python
def _load_checkpoint(self):
    """Load Actor, Critic, and DataLoader state from checkpoint."""
```

**核心职责：**
- 根据 resume_mode 决定加载策略
- 加载 Actor 和 Critic 模型
- 加载 DataLoader 状态

### 1.5 代码详解

```python
def _load_checkpoint(self):
    if self.config.trainer.resume_mode == "disable":
        return 0
    
    checkpoint_folder = self.config.trainer.default_local_dir
    if not os.path.isabs(checkpoint_folder):
        working_dir = os.getcwd()
        checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
    
    global_step_folder = find_latest_ckpt_path(checkpoint_folder)
    
    if self.config.trainer.resume_mode == "auto":
        if global_step_folder is None:
            print("Training from scratch")
            return 0
    else:
        if self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
    
    # set global step
    self.global_steps = int(global_step_folder.split("global_step_")[-1])
    
    actor_path = os.path.join(global_step_folder, "actor")
    self.actor_rollout_wg.load_checkpoint(actor_path)
    
    if self.use_critic:
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        self.critic_wg.load_checkpoint(critic_path)
    
    dataloader_local_path = os.path.join(global_step_folder, "data.pt")
    if os.path.exists(dataloader_local_path):
        dataloader_state_dict = torch.load(dataloader_local_path)
        self.train_dataloader.load_state_dict(dataloader_state_dict)
```

### 1.6 resume_mode 配置

```
┌─────────────────────────────────────────────────────────────────────┐
│                  resume_mode 配置                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  resume_mode = "disable":                                           │
│  ─────────────────────                                              │
│  - 不加载任何 checkpoint                                            │
│  - 从头开始训练                                                    │
│                                                                      │
│  resume_mode = "auto":                                              │
│  ────────────────                                                   │
│  - 自动查找最新 checkpoint                                          │
│  - 如果找到则加载                                                  │
│  - 如果没找到则从头开始                                            │
│                                                                      │
│  resume_mode = "resume_path":                                       │
│  ─────────────────────                                              │
│  - 从指定路径加载 checkpoint                                        │
│  - 需要配置 resume_from_path                                       │
│                                                                      │
│  配置示例：                                                          │
│  ─────────                                                          │
│                                                                      │
│  # 自动恢复                                                          │
│  trainer:                                                            │
│    resume_mode: "auto"                                              │
│    default_local_dir: "./checkpoints"                              │
│                                                                      │
│  # 指定路径恢复                                                      │
│  trainer:                                                            │
│    resume_mode: "resume_path"                                       │
│    resume_from_path: "./checkpoints/global_step_500"               │
│                                                                      │
│  # 不恢复                                                            │
│  trainer:                                                            │
│    resume_mode: "disable"                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、_validate 方法

### 2.1 方法定义

```python
def _validate(self):
    """Evaluate model on validation dataset."""
```

**核心职责：**
- 在验证集上评估模型性能
- 记录验证指标
- 记录生成样本

### 2.2 代码详解

```python
def _validate(self):
    data_source_lst = []
    reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    
    sample_inputs = []
    sample_outputs = []
    sample_scores = []
    
    for test_data in self.val_dataloader:
        test_batch = DataProto.from_single_dict(test_data)
        
        # add uid
        if "uid" not in test_batch.non_tensor_batch:
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))])
        
        # repeat test batch
        test_batch = test_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
        )
        
        # skip if reward model style is "model"
        if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            return {}
        
        test_gen_batch = self._get_gen_batch(test_batch)
        test_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }
        
        # generate sequences
        test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size)
        
        test_batch = test_batch.union(test_output_gen_batch)
        
        # compute reward
        result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
        scores = result["reward_tensor"].sum(-1).cpu().tolist()
        sample_scores.extend(scores)
        
        reward_extra_infos_dict["reward"].extend(scores)
    
    # log generations
    self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
    
    # process metrics
    metric_dict = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
    
    return metric_dict
```

### 2.3 验证流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                  验证流程                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  val_kwargs.n:                                                      │
│  ─────────                                                          │
│  验证时每个 prompt 生成多少个 response                              │
│                                                                      │
│  示例：                                                              │
│  val_kwargs.n = 8                                                   │
│  → 每个 prompt 生成 8 个 response                                  │
│  → 计算 8 个 response 的 reward                                   │
│  → 可以统计 best@8, maj@8 等                                      │
│                                                                      │
│  val_kwargs.do_sample:                                              │
│  ───────────────────                                                │
│  True: 采样生成（有随机性）                                         │
│  False: greedy 生成（确定性）                                      │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  验证指标：                                                          │
│  ─────────                                                          │
│                                                                      │
│  mean@N: N 个 response 的平均 reward                               │
│  best@N: N 个 response 中最好的 reward                             │
│  maj@N: N 个 response 中多数投票的结果                             │
│                                                                      │
│  计算方式：                                                          │
│  ─────────                                                          │
│                                                                      │
│  同一 prompt 的 N 个 response:                                      │
│  [r_1, r_2, r_3, ..., r_N]                                         │
│                                                                      │
│  mean@N = mean([r_1, ..., r_N])                                    │
│  best@N = max([r_1, ..., r_N])                                     │
│  maj@N = vote([r_1, ..., r_N])                                     │
│                                                                      │
│  用途：                                                              │
│  ─────                                                              │
│  mean@N: 平均性能                                                   │
│  best@N: 最佳性能潜力                                               │
│  maj@N: 多数投票性能（稳定性）                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.4 _maybe_log_val_generations 方法

```python
def _maybe_log_val_generations(self, inputs, outputs, scores):
    """Log a table of validation samples to the configured logger."""
    
    generations_to_log = self.config.trainer.log_val_generations
    
    if generations_to_log == 0:
        return
    
    # Create tuples and sort
    samples = list(zip(inputs, outputs, scores))
    samples.sort(key=lambda x: x[0])
    
    # Shuffle with fixed seed
    rng = np.random.RandomState(42)
    rng.shuffle(samples)
    
    # Take first N
    samples = samples[:generations_to_log]
    
    # Log to logger
    self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
```

**log_val_generations 配置：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  log_val_generations 配置                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  log_val_generations = N:                                           │
│  ─────────────────────                                              │
│  记录 N 个验证样本到 wandb/swanlab                                 │
│                                                                      │
│  N = 0:                                                             │
│  ─────                                                              │
│  不记录任何样本                                                     │
│                                                                      │
│  N > 0:                                                             │
│  ─────                                                              │
│  记录 N 个样本的 input、output、score                              │
│  用于可视化检查                                                     │
│                                                                      │
│  示例：                                                              │
│  ─────                                                              │
│                                                                      │
│  trainer:                                                            │
│    log_val_generations: 10                                          │
│                                                                      │
│  结果：                                                              │
│  wandb 上显示一个表格                                               │
│  包含 10 个验证样本                                                 │
│  可以看到 input、生成的 output、score                             │
│                                                                      │
│  用途：                                                              │
│  ─────                                                              │
│  - 人工检查生成质量                                                 │
│  - 发现训练问题                                                     │
│  - 对比不同 step 的生成变化                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.5 _dump_generations 方法

```python
def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
    """Dump rollout/validation samples as JSONL."""
    
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
    
    n = len(inputs)
    base_data = {
        "input": inputs,
        "output": outputs,
        "gts": gts,
        "score": scores,
        "step": [self.global_steps] * n,
    }
    
    for k, v in reward_extra_infos_dict.items():
        if len(v) == n:
            base_data[k] = v
    
    lines = []
    for i in range(n):
        entry = {k: v[i] for k, v in base_data.items()}
        lines.append(json.dumps(entry, ensure_ascii=False))
    
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
```

**JSONL 文件格式：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  JSONL 文件格式                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  文件名: {global_steps}.jsonl                                       │
│                                                                      │
│  内容示例：                                                          │
│  ─────────                                                          │
│                                                                      │
│  {"input": "请解释PPO", "output": "PPO是...", "gts": null,         │
│   "score": 0.85, "step": 100}                                       │
│                                                                      │
│  {"input": "写排序算法", "output": "def sort...", "gts": "...",    │
│   "score": 1.0, "step": 100}                                        │
│                                                                      │
│  每行一个 JSON 对象                                                  │
│                                                                      │
│  用途：                                                              │
│  ─────                                                              │
│  - 记录每个 step 的生成样本                                        │
│  - 用于后续分析                                                     │
│  - 对比不同 step 的生成变化                                        │
│                                                                      │
│  配置：                                                              │
│  ─────                                                              │
│                                                                      │
│  trainer:                                                            │
│    rollout_data_dir: "./rollout_data"    # 训练时记录              │
│    validation_data_dir: "./val_data"     # 验证时记录              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、调用时机

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Checkpoint 和验证调用时机                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  fit() 训练循环中：                                                  │
│  ─────────                                                          │
│                                                                      │
│  初始验证：                                                          │
│  ─────────                                                          │
│  if val_before_train:                                               │
│      _validate()                                                    │
│                                                                      │
│  定期验证：                                                          │
│  ─────────                                                          │
│  if global_steps % test_freq == 0:                                  │
│      _validate()                                                    │
│                                                                      │
│  定期保存：                                                          │
│  ─────────                                                          │
│  if global_steps % save_freq == 0:                                  │
│      _save_checkpoint()                                             │
│                                                                      │
│  训练结束：                                                          │
│  ─────────                                                          │
│  if is_last_step:                                                   │
│      _validate()                                                    │
│      _save_checkpoint()                                             │
│                                                                      │
│  ESI 预期到期：                                                      │
│  ─────────                                                          │
│  if esi_close_to_expiration:                                        │
│      _save_checkpoint()  # 强制保存                                 │
│                                                                      │
│  配置示例：                                                          │
│  ─────────                                                          │
│                                                                      │
│  trainer:                                                            │
│    test_freq: 100           # 每 100 steps 验证一次                │
│    save_freq: 500           # 每 500 steps 保存一次                │
│    val_before_train: true   # 训练前先验证                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、总结

| 方法 | 职责 | 关键配置 |
|------|------|----------|
| `_save_checkpoint` | 保存模型和 DataLoader | `save_freq`, `max_*_ckpt_to_keep` |
| `_load_checkpoint` | 加载模型恢复训练 | `resume_mode`, `resume_from_path` |
| `_validate` | 验证集评估 | `test_freq`, `val_kwargs.n` |
| `_maybe_log_val_generations` | 记录验证样本到 wandb | `log_val_generations` |
| `_dump_generations` | 保存样本到 JSONL | `rollout_data_dir`, `validation_data_dir` |

**一句话总结：**
Checkpoint 和验证方法实现了训练恢复、定期评估和样本记录功能，支持灵活的 resume_mode 和验证配置。