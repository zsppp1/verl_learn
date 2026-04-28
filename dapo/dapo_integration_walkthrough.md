# DAPO 算法对接 verl 框架解析

> verl/recipe/dapo/main_dapo.py 和 dapo_ray_trainer.py

## 一、DAPO 对接 verl 的整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DAPO 与 verl 框架对接关系                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  verl 核心框架                                                       │
│  ─────────────                                                      │
│  ├── main_ppo.py              # 标准 PPO 入口                       │
│  ├── ray_trainer.py           # RayPPOTrainer 基类                  │
│  ├── core_algos.py            # 算法实现（GAE, KL等）                │
│  └── workers/                 # Worker 实现                         │
│                                                                      │
│  DAPO 扩展（继承 verl）                                              │
│  ───────────────────                                                │
│  ├── main_dapo.py             # DAPO 专用入口                       │
│  │   └── TaskRunner           # 与 main_ppo.py 类似结构            │
│  │       └── RayDAPOTrainer   # 关键：继承 RayPPOTrainer           │
│  │                                                                  │
│  ├── dapo_ray_trainer.py      # DAPO 核心实现                       │
│  │   └── RayDAPOTrainer       # 继承 + 重写 fit()                  │
│  │                                                                  │
│  └── config/                                                        │
│      └── dapo_trainer.yaml    # DAPO 配置                           │
│                                                                      │
│  对接方式                                                            │
│  ───────                                                            │
│  1. 继承 RayPPOTrainer                                              │
│  2. 重写 fit() 方法（添加动态采样）                                  │
│  3. 使用 verl 原有组件（Worker, ResourcePool, compute_advantage等） │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、main_dapo.py 解析

### 2.1 文件结构

```python
# main_dapo.py 结构（与 main_ppo.py 类似）

@hydra.main(config_path="config", config_name="dapo_trainer")
def main(config):
    auto_set_device(config)
    run_ppo(config)

def run_ppo(config):
    # 初始化 Ray
    ray.init(...)
    
    # 创建 TaskRunner Actor
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        # 加载模型、tokenizer
        # 创建 role_worker_mapping
        # 创建 ResourcePoolManager
        # 创建 RayDAPOTrainer（而非 RayPPOTrainer）
        trainer = RayDAPOTrainer(...)
        trainer.init_workers()
        trainer.fit()
```

### 2.2 与 main_ppo.py 的关键差异

| 差异点 | main_ppo.py | main_dapo.py |
|--------|-------------|--------------|
| Trainer 类 | `RayPPOTrainer` | `RayDAPOTrainer` |
| 配置文件 | `ppo_trainer.yaml` | `dapo_trainer.yaml` |
| reward_fn | 标准 load | 带 `overlong_buffer_cfg` 参数 |

### 2.3 DAPO 特有的 reward_fn 加载

```python
# main_dapo.py 第152-167行
reward_fn = load_reward_manager(
    config,
    tokenizer,
    0,
    max_resp_len=config.data.max_response_length,
    overlong_buffer_cfg=config.reward_model.overlong_buffer,  # DAPO 特有
)
```

**overlong_buffer 是 DAPO 的超长惩罚配置：**

```yaml
reward_model:
  overlong_buffer:
    enable: true
    len: 4096        # 缓冲区长度
    penalty_factor: 1.0  # 惩罚系数
```

---

## 三、dapo_ray_trainer.py 解析

### 3.1 RayDAPOTrainer 类定义

```python
class RayDAPOTrainer(RayPPOTrainer):
    """
    继承 RayPPOTrainer，重写 fit() 方法实现 DAPO 特有逻辑
    """
```

**继承关系：**

```
RayPPOTrainer (verl 核心)
    │
    │  提供：
    │  - init_workers()
    │  - _validate()
    │  - _save_checkpoint()
    │  - _load_checkpoint()
    │  - _compute_values()
    │  - compute_kl_related_metrics()
    │  - WorkerGroup 管理
    │
    ↓
RayDAPOTrainer (DAPO 扩展)
    │
    │  重写：
    │  - fit()  ← 添加动态采样逻辑
    │
    │  新增：
    │  - compute_kl_related_metrics()（优化）
    │
```

---

### 3.2 fit() 方法详解 - DAPO 核心逻辑

**整体流程：**

```
fit()
│
│  初始化
│  ─────────────────────────────────────────────────────────────
│  logger = Tracking(...)
│  self.global_steps = 0
│  self.gen_steps = 0
│  self._load_checkpoint()
│  self._validate()  # 初始验证
│
│  主循环
│  ─────────────────────────────────────────────────────────────
│  for epoch in range(total_epochs):
│      for batch_dict in train_dataloader:
│          │
│          │  ┌─────────────────────────────────────────────────┐
│          │  │         DAPO 动态采样循环                        │
│          │  │                                                 │
│          │  │  while num_prompt_in_batch < train_batch_size: │
│          │  │      │                                          │
│          │  │      │  generate_sequences()                   │
│          │  │      │  compute_reward()                       │
│          │  │      │                                          │
│          │  │      │  filter_groups:                         │
│          │  │      │  ───────────────────────────────────   │
│          │  │      │  if std(rewards) == 0:                  │
│          │  │      │      # 全对或全错，过滤掉                │
│          │  │      │      continue  # 重新采样                │
│          │  │      │  else:                                  │
│          │  │      │      # 保留有效样本                      │
│          │  │      │      batch.add(new_batch)               │
│          │  │      │                                          │
│          │  │      │  if num_gen_batches >= max_limit:      │
│          │  │      │      raise Error                        │
│          │  │      │                                          │
│          │  └─────────────────────────────────────────────────┘
│          │
│          │  标准 PPO 流程（与 RayPPOTrainer 相同）
│          │  ─────────────────────────────────────────────────────
│          │  compute_kl_related_metrics()
│          │  compute_values()
│          │  compute_advantage()
│          │  update_critic()
│          │  update_actor()
│          │
│          │  验证 + 保存 checkpoint
│          │
│          global_steps += 1
```

---

### 3.3 动态采样核心代码

```python
# dapo_ray_trainer.py 第229-288行

if not self.config.algorithm.filter_groups.enable:
    # 不启用动态采样：直接使用
    batch = new_batch
else:
    # 启用动态采样：DAPO 核心
    metric_name = self.config.algorithm.filter_groups.metric
    
    # 计算每个 prompt 的所有 response 的 reward
    prompt_uid2metric_vals = defaultdict(list)
    for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], 
                                new_batch.non_tensor_batch[metric_name]):
        prompt_uid2metric_vals[uid].append(metric_val)
    
    # 计算 std，判断是否全对/全错
    prompt_uid2metric_std = {}
    for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
        prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
    
    # 过滤：只保留 std > 0 的 prompt（有差异的）
    kept_prompt_uids = [
        uid for uid, std in prompt_uid2metric_std.items()
        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
    ]
    
    num_prompt_in_batch += len(kept_prompt_uids)
    
    # 检查是否需要继续采样
    prompt_bsz = self.config.data.train_batch_size
    if num_prompt_in_batch < prompt_bsz:
        # 样本不够，继续采样
        print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
        max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
        
        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
            self.gen_steps += 1
            continue  # ← 关键：跳过后续，重新采样
        else:
            raise ValueError("Generated too many. Data too difficult?")
    else:
        # 样本够了，对齐 batch
        traj_bsz = train_batch_size * rollout.n
        batch = batch[:traj_bsz]
```

---

### 3.4 动态采样流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DAPO 动态采样流程                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  生成阶段                                                            │
│  ───────                                                            │
│                                                                      │
│  gen_batch (N prompts)                                               │
│      │                                                               │
│      │ generate_sequences (每个 prompt 生成 n 个 response)          │
│      │                                                               │
│      ↓                                                               │
│  gen_batch_output (N × n responses)                                  │
│      │                                                               │
│      │ compute_reward                                                │
│      │                                                               │
│      ↓                                                               │
│  rewards for each response                                           │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  过滤阶段                                                            │
│  ───────                                                            │
│                                                                      │
│  对每个 prompt 的 n 个 response 计算 reward std:                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  prompt_1: [r1=1.0, r2=1.0, r3=1.0, r4=1.0]                │   │
│  │            std = 0 → 全对 → 过滤掉                          │   │
│  │                                                             │   │
│  │  prompt_2: [r1=0.0, r2=0.0, r3=0.0, r4=0.0]                │   │
│  │            std = 0 → 全错 → 过滤掉                          │   │
│  │                                                             │   │
│  │  prompt_3: [r1=1.0, r2=0.0, r3=1.0, r4=0.5]                │   │
│  │            std > 0 → 有差异 → 保留                          │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  循环判断                                                            │
│  ───────                                                            │
│                                                                      │
│  if num_valid_prompts < train_batch_size:                           │
│      continue  # 重新生成下一个 batch                               │
│                                                                      │
│  if num_gen_batches >= max_num_gen_batches:                         │
│      raise Error  # 生成次数超限                                    │
│                                                                      │
│  else:                                                              │
│      # 样本够了，开始训练                                            │
│      compute_advantage()                                            │
│      update_actor()                                                 │
│      update_critic()                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、DAPO 对接 verl 的关键点

### 4.1 继承复用

| verl 组件 | DAPO 使用方式 |
|-----------|---------------|
| `RayPPOTrainer` | 继承，复用 init_workers, _validate, _save_checkpoint |
| `WorkerGroup` | 直接使用，不做修改 |
| `ResourcePoolManager` | 直接使用 |
| `compute_advantage()` | 直接调用 |
| `apply_kl_penalty()` | 直接调用 |
| `agg_loss()` | 直接调用 |

### 4.2 重写扩展

| 组件 | DAPO 改动 |
|------|-----------|
| `fit()` | 重写，添加动态采样循环 |
| `compute_kl_related_metrics()` | 小优化 |
| `reward_fn` | 添加 overlong_buffer 参数 |

### 4.3 配置扩展

```yaml
# dapo_trainer.yaml

algorithm:
  filter_groups:             # DAPO 特有配置
    enable: true
    metric: acc
    max_num_gen_batches: 10

reward_model:
  overlong_buffer:           # DAPO 特有配置
    enable: true
    len: 4096
    penalty_factor: 1.0

actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2      # DAPO 不对称裁剪
    clip_ratio_high: 0.28
    loss_agg_mode: "token-mean"
```

---

## 五、DAPO 四大创新在 verl 中的实现位置

| 创新 | 配置 | 代码位置 |
|------|------|----------|
| **不对称裁剪** | `clip_ratio_low/clip_ratio_high` | `core_algos.py` compute_policy_loss |
| **动态采样** | `filter_groups.enable` | `dapo_ray_trainer.py` fit() 第229-288行 |
| **Token-level Loss** | `loss_agg_mode: token-mean` | `core_algos.py` agg_loss() |
| **超长惩罚** | `overlong_buffer` | `reward_manager/dapo.py` |

---

## 六、与标准 PPO 的代码流程对比

### 6.1 main_ppo.py 流程

```
main_ppo.py
    │
    │  TaskRunner.run()
    │      │
    │      │  add_actor_rollout_worker()
    │      │  add_critic_worker()
    │      │  add_reward_model_worker()
    │      │  add_ref_policy_worker()
    │      │
    │      │  RayPPOTrainer()
    │      │      │
    │      │      │  init_workers()
    │      │      │  fit()
    │      │      │      │
    │      │      │      │  for batch in dataloader:
    │      │      │      │      generate()
    │      │      │      │      compute_reward()
    │      │      │      │      compute_advantage()
    │      │      │      │      update_actor()
    │      │      │      │      update_critic()
    │      │      │      │
```

### 6.2 main_dapo.py 流程

```
main_dapo.py
    │
    │  TaskRunner.run()
    │      │
    │      │  (简化版，直接创建 role_worker_mapping)
    │      │  RayDAPOTrainer()  ← 继承 RayPPOTrainer
    │      │      │
    │      │      │  init_workers()  ← 继承，不修改
    │      │      │  fit()           ← 重写
    │      │      │      │
    │      │      │      │  for batch in dataloader:
    │      │      │      │      │
    │      │      │      │      │  while samples_not_enough:  ← DAPO 特有循环
    │      │      │      │      │      generate()
    │      │      │      │      │      compute_reward()
    │      │      │      │      │      filter_groups()  ← 过滤
    │      │      │      │      │      if filtered_out: continue
    │      │      │      │      │
    │      │      │      │      │  compute_advantage()
    │      │      │      │      │  update_actor()
    │      │      │      │      │  update_critic()
```

---

## 七、总结

| 方面 | 内容 |
|------|------|
| **对接方式** | 继承 `RayPPOTrainer` + 重写 `fit()` |
| **复用 verl** | WorkerGroup, ResourcePool, compute_advantage 等全部复用 |
| **DAPO 特有** | 动态采样循环、overlong_buffer、不对称裁剪配置 |
| **入口文件** | `main_dapo.py`（简化版） |
| **核心文件** | `dapo_ray_trainer.py`（RayDAPOTrainer） |

**一句话总结：**
DAPO 通过继承 `RayPPOTrainer` 复用 verl 的分布式训练架构，仅在 `fit()` 方法中添加动态采样循环，实现 DAPO 算法的核心逻辑。