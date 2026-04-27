# TaskRunner.run() 方法解析

> verl/trainer/main_ppo.py 第256-362行

## 一、方法定义

```python
def run(self, config):
    """Execute the main PPO training workflow.

    This method sets up the distributed training environment, initializes
    workers, datasets, and reward functions, then starts the training process.

    Args:
        config: Training configuration object containing all parameters needed
               for setting up and running the PPO training process.
    """
```

**核心职责：**
- 打印和解析配置
- 添加所有 Worker（Actor、Critic、Reward、RefPolicy）
- 验证配置有效性
- 加载模型和 Tokenizer
- 创建 Reward Manager
- 初始化资源池
- 创建数据集和 Sampler
- 创建 RayPPOTrainer
- 启动训练流程

---

## 二、整体流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TaskRunner.run() 整体流程                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 阶段1: 配置初始化                                              │  │
│  │                                                               │  │
│  │  ① 打印 hostname 和 PID                                       │  │
│  │  ② pprint 配置内容                                            │  │
│  │  ③ OmegaConf.resolve(config)                                 │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 阶段2: Worker 添加                                            │  │
│  │                                                               │  │
│  │  ④ add_actor_rollout_worker(config)                          │  │
│  │  ⑤ add_critic_worker(config)                                 │  │
│  │  ⑥ add_reward_model_worker(config)                           │  │
│  │  ⑦ add_ref_policy_worker(config, actor_rollout_cls)          │  │
│  │                                                               │  │
│  │  结果: role_worker_mapping + mapping                          │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 阶段3: 配置验证                                               │  │
│  │                                                               │  │
│  │  ⑧ validate_config(config, use_reference_policy, use_critic) │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 阶段4: 模型加载                                               │  │
│  │                                                               │  │
│  │  ⑨ copy_to_local(model.path) → 本地路径                       │  │
│  │  ⑩ hf_tokenizer(local_path) → Tokenizer                      │  │
│  │  ⑪ hf_processor(local_path) → Processor (可选)               │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 阶段5: Reward Manager                                         │  │
│  │                                                               │  │
│  │  ⑫ load_reward_manager(config) → reward_fn                   │  │
│  │  ⑬ load_reward_manager(config) → val_reward_fn               │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 阶段6: 资源池和数据集                                         │  │
│  │                                                               │  │
│  │  ⑭ init_resource_pool_mgr(config) → resource_pool_manager    │  │
│  │  ⑮ create_rl_dataset(train_files) → train_dataset            │  │
│  │  ⑯ create_rl_dataset(val_files) → val_dataset                │  │
│  │  ⑳ create_rl_sampler(config) → train_sampler                 │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 阶段7: Trainer 创建和训练                                     │  │
│  │                                                               │  │
│  │  ㉑ RayPPOTrainer(...) → trainer                             │  │
│  │  ㉒ trainer.init_workers()                                   │  │
│  │  ㉓ trainer.fit()                                            │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、详细代码解析

### 3.1 阶段1: 配置初始化

```python
# Print the initial configuration. `resolve=True` will evaluate symbolic values.
from pprint import pprint
from omegaconf import OmegaConf
from verl.utils.fs import copy_to_local

print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
pprint(OmegaConf.to_container(config, resolve=True))
OmegaConf.resolve(config)
```

**各步骤作用：**

| 步骤 | 作用 |
|------|------|
| `print(hostname, PID)` | 调试信息，确认运行节点 |
| `pprint(config)` | 打印完整配置，便于排查问题 |
| `OmegaConf.resolve(config)` | 解析配置中的符号引用，如 `${trainer.nnodes}` |

**OmegaConf.resolve 示例：**

```yaml
# 配置文件中的符号引用
trainer:
  nnodes: 2
  total_gpus: ${trainer.nnodes} * ${trainer.n_gpus_per_node}

# resolve 后
trainer:
  nnodes: 2
  total_gpus: 16  # 已计算
```

---

### 3.2 阶段2: Worker 添加

```python
actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
self.add_critic_worker(config)

# We should adopt a multi-source reward function here:
# - for rule-based rm, we directly call a reward score
# - for model-based rm, we call a model
# - for code related prompt, we send to a sandbox if there are test cases
# finally, we combine all the rewards together
# The reward type depends on the tag of the data
self.add_reward_model_worker(config)

# Add a reference policy worker if KL loss or KL reward is used.
self.add_ref_policy_worker(config, actor_rollout_cls)
```

**源码注释解读：**

> "We should adopt a multi-source reward function here..."

**多源奖励系统的设计：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  多源奖励系统架构                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Reward 来源类型：                                                   │
│  ─────────────                                                      │
│                                                                      │
│  1. Rule-based Reward (规则奖励):                                   │
│     ──────────────────────────────                                  │
│     - 直接调用评分函数                                               │
│     - 例如：代码执行结果、数学答案正确性                             │
│                                                                      │
│  2. Model-based Reward (模型奖励):                                  │
│     ──────────────────────────────                                  │
│     - 调用 Reward Model                                             │
│     - 例如：人类偏好评分模型                                         │
│                                                                      │
│  3. Sandbox Reward (沙箱奖励):                                      │
│     ──────────────────────────────                                  │
│     - 发送到沙箱执行测试                                             │
│     - 例如：代码运行结果、API调用结果                                │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   Reward 计算流程：                                            │  │
│  │                                                               │  │
│  │   Prompt + Response                                           │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │              根据数据标签选择奖励来源                   │   │  │
│  │   │                                                       │   │  │
│  │   │   tag = "code" → Sandbox (执行代码)                   │   │  │
│  │   │   tag = "math" → Rule-based (计算正确性)              │   │  │
│  │   │   tag = "chat" → Model-based (偏好模型)               │   │  │
│  │   │                                                       │   │  │
│  │   └──────────────────────────────────────────────────────┘   │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │   ┌──────────────────────────────────────────────────────┐   │  │
│  │   │              合成最终奖励                               │   │  │
│  │   │                                                       │   │  │
│  │   │   reward_final = Σ reward_i * weight_i                │   │  │
│  │   │                                                       │   │  │
│  │   └──────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Worker 添加顺序的意义：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Worker 添加顺序分析                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  顺序：                                                              │
│  ─────                                                              │
│  1. ActorRollout Worker (返回 actor_rollout_cls)                   │
│  2. Critic Worker                                                   │
│  3. Reward Model Worker                                             │
│  4. Ref Policy Worker (使用 actor_rollout_cls)                     │
│                                                                      │
│  为什么 Ref Policy 在最后？                                          │
│  ─────────────────────────                                           │
│  - Ref Policy 需要 actor_rollout_cls 参数                          │
│  - actor_rollout_cls 来自第一步的返回值                             │
│  - Actor 和 Ref Policy 共享同一个模型                               │
│                                                                      │
│  为什么 Reward Model 在 Critic 之后？                                │
│  ─────────────────────────────                                       │
│  - Reward Model 可选（enable=False 则不添加）                       │
│  - Critic 是必需的（PPO 必须有 Value Function）                     │
│                                                                      │
│  mapping 的构建顺序：                                                │
│  ───────────────────                                                │
│  添加顺序 → mapping 顺序 → ResourcePoolManager 顺序                 │
│                                                                      │
│  Role → Pool 映射：                                                 │
│  ActorRollout → global_pool                                        │
│  Critic → global_pool                                              │
│  RewardModel → reward_pool (可选)                                  │
│  RefPolicy → global_pool                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.3 阶段3: 配置验证

```python
# validate config
validate_config(
    config=config,
    use_reference_policy=need_reference_policy(self.role_worker_mapping),
    use_critic=need_critic(config),
)
```

**need_reference_policy 函数：**

```python
def need_reference_policy(role_worker_mapping):
    """检查是否需要 Reference Policy"""
    return Role.RefPolicy in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping
```

**need_critic 函数：**

```python
def need_critic(config):
    """检查是否需要 Critic"""
    return config.critic.enable
```

**validate_config 检查内容：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  validate_config 检查项                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 配置一致性检查：                                                 │
│     ──────────────                                                  │
│     - 如果 use_reference_policy=True 但未添加 RefPolicy Worker       │
│     - 如果 use_critic=True 但未添加 Critic Worker                   │
│                                                                      │
│  2. 参数有效性检查：                                                 │
│     ──────────────                                                  │
│     - KL 惩罚系数是否为正数                                          │
│     - GAE lambda 是否在 [0, 1] 范围                                 │
│     - PPO clip epsilon 是否合理                                     │
│                                                                      │
│  3. 依赖关系检查：                                                   │
│     ──────────────                                                  │
│     - 使用 KL 约束时必须有 Reference Policy                         │
│     - 使用 GAE 时必须有 Critic                                       │
│                                                                      │
│  错误处理：                                                          │
│     ────────                                                        │
│     - 配置不一致 → raise ValueError                                 │
│     - 参数无效 → raise ValueError                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.4 阶段4: 模型加载

```python
# Download the checkpoint from HDFS to the local machine.
# `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
local_path = copy_to_local(
    config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
)

# Instantiate the tokenizer and processor.
from verl.utils import hf_processor, hf_tokenizer

trust_remote_code = config.data.get("trust_remote_code", False)
tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
# Used for multimodal LLM, could be None
processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
```

**copy_to_local 详解：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  copy_to_local 功能                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  源路径类型：                                                        │
│  ─────────                                                          │
│                                                                      │
│  1. HDFS 路径:                                                       │
│     hdfs://cluster/path/to/model                                    │
│     → 从 HDFS 下载到本地                                             │
│                                                                      │
│  2. HTTP URL:                                                        │
│     https://huggingface.co/model                                    │
│     → 从网络下载到本地                                               │
│                                                                      │
│  3. 本地路径:                                                        │
│     /local/path/to/model                                            │
│     → 直接返回，不复制                                               │
│                                                                      │
│  use_shm 参数：                                                      │
│  ─────────────                                                      │
│                                                                      │
│  True:                                                               │
│  - 使用共享内存 (/dev/shm)                                          │
│  - 多 Worker 可以共享同一份模型内存                                  │
│  - 加载速度更快                                                      │
│  - 适合多卡分布式训练                                                │
│                                                                      │
│  False:                                                              │
│  - 使用普通文件系统                                                  │
│  - 每个 Worker 独立加载                                              │
│  - 占用更多磁盘空间                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Tokenizer 和 Processor：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Tokenizer 和 Processor                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Tokenizer:                                                          │
│  ─────────                                                           │
│  - 文本 → Token ID 序列                                              │
│  - 所有 LLM 必需                                                     │
│                                                                      │
│  Processor:                                                          │
│  ─────────                                                           │
│  - 多模态 LLM 专用                                                   │
│  - 处理图像、音频等输入                                              │
│  - 单模态 LLM 时为 None                                              │
│                                                                      │
│  trust_remote_code:                                                  │
│  ──────────────────                                                  │
│  - 是否信任远程代码                                                  │
│  - 一些模型需要执行仓库中的自定义代码                                │
│  - 例如：Qwen、ChatGLM 的自定义 tokenizer                           │
│                                                                      │
│  示例：                                                              │
│  ─────                                                              │
│                                                                      │
│  文本 LLM (如 Llama):                                                │
│  tokenizer = LlamaTokenizer(...)                                    │
│  processor = None                                                    │
│                                                                      │
│  多模态 LLM (如 LLaVA):                                              │
│  tokenizer = LlamaTokenizer(...)                                    │
│  processor = LlavaProcessor(...)  # 处理图像输入                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.5 阶段5: Reward Manager

```python
# Load the reward manager for training and validation.
reward_fn = load_reward_manager(
    config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
)
val_reward_fn = load_reward_manager(
    config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
)
```

**num_examine 参数差异：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  num_examine 参数解释                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  reward_fn (训练用):                                                 │
│  ───────────────────                                                 │
│  num_examine = 0                                                     │
│  - 不打印任何样本                                                    │
│  - 避免训练时的日志开销                                              │
│                                                                      │
│  val_reward_fn (验证用):                                             │
│  ─────────────────────                                               │
│  num_examine = 1                                                     │
│  - 打印第一个样本的详细信息                                          │
│  - 用于调试和检查奖励计算                                            │
│  - 验证奖励函数是否正常工作                                          │
│                                                                      │
│  reward_kwargs:                                                      │
│  ─────────────                                                       │
│  - 传递给 Reward Manager 的额外参数                                  │
│  - 例如：模型路径、评分阈值等                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.6 阶段6: 资源池和数据集

```python
resource_pool_manager = self.init_resource_pool_mgr(config)

from verl.utils.dataset.rl_dataset import collate_fn

# Create training and validation datasets.
train_dataset = create_rl_dataset(
    config.data.train_files,
    config.data,
    tokenizer,
    processor,
    is_train=True,
    max_samples=config.data.get("train_max_samples", -1),
)
val_dataset = create_rl_dataset(
    config.data.val_files,
    config.data,
    tokenizer,
    processor,
    is_train=False,
    max_samples=config.data.get("val_max_samples", -1),
)
train_sampler = create_rl_sampler(config.data, train_dataset)
```

**数据集创建流程：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  数据集创建流程                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入文件：                                                          │
│  ─────────                                                          │
│  train_files: ["train_1.jsonl", "train_2.jsonl"]                    │
│  val_files: ["val.jsonl"]                                           │
│                                                                      │
│  数据格式：                                                          │
│  ─────────                                                          │
│  {"prompt": "请解释什么是PPO", "responses": [...]}                  │
│                                                                      │
│  create_rl_dataset 步骤：                                            │
│  ─────────────────────                                              │
│  1. get_dataset_class(data_config) → Dataset类                      │
│  2. dataset_cls(data_files, tokenizer, ...) → Dataset实例           │
│                                                                      │
│  max_samples 参数：                                                  │
│  ──────────────────                                                 │
│  - -1: 使用全部数据                                                  │
│  - N: 只使用前 N 个样本（用于调试）                                  │
│                                                                      │
│  collate_fn:                                                        │
│  ─────────                                                          │
│  - 将多个样本合并为一个 batch                                        │
│  - 处理 padding、attention mask 等                                  │
│                                                                      │
│  sampler:                                                           │
│  ────────                                                           │
│  - RandomSampler: 随机采样（shuffle=True）                          │
│  - SequentialSampler: 顺序采样（shuffle=False）                     │
│  - CurriculumSampler: 课程学习采样（可选）                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.7 阶段7: Trainer 创建和训练

```python
# Initialize the PPO trainer.
trainer = RayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    processor=processor,
    role_worker_mapping=self.role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    ray_worker_group_cls=ray_worker_group_cls,
    reward_fn=reward_fn,
    val_reward_fn=val_reward_fn,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    collate_fn=collate_fn,
    train_sampler=train_sampler,
)

# Initialize the workers of the trainer.
trainer.init_workers()

# Start the training process.
trainer.fit()
```

**RayPPOTrainer 参数详解：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  RayPPOTrainer 参数                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  核心参数：                                                          │
│  ─────────                                                          │
│                                                                      │
│  config:                                                             │
│  - 完整配置对象                                                      │
│  - 包含所有训练参数                                                  │
│                                                                      │
│  tokenizer + processor:                                             │
│  - 文本和多模态处理                                                  │
│                                                                      │
│  role_worker_mapping:                                               │
│  - Role → Worker 类映射                                             │
│  - 例如：{ActorRollout: AsyncActorRolloutRefWorker}                 │
│                                                                      │
│  resource_pool_manager:                                             │
│  - GPU 资源池管理                                                   │
│  - 分配 Worker 到对应 GPU                                           │
│                                                                      │
│  ray_worker_group_cls:                                              │
│  - WorkerGroup 类                                                   │
│  - 用于创建 Worker 集合                                             │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  数据相关参数：                                                      │
│  ─────────                                                          │
│                                                                      │
│  reward_fn + val_reward_fn:                                         │
│  - 训练和验证的奖励函数                                              │
│                                                                      │
│  train_dataset + val_dataset:                                       │
│  - 训练和验证数据集                                                  │
│                                                                      │
│  collate_fn:                                                        │
│  - batch 合并函数                                                   │
│                                                                      │
│  train_sampler:                                                     │
│  - 训练数据采样器                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**trainer.init_workers() 流程：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  trainer.init_workers() 流程                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  步骤：                                                              │
│  ─────                                                              │
│                                                                      │
│  1. 获取资源池：                                                     │
│     for role, pool_id in mapping.items():                          │
│         pool = resource_pool_manager.get_resource_pool(role)        │
│                                                                      │
│  2. 创建 WorkerGroup：                                               │
│     worker_group = ray_worker_group_cls(                            │
│         resource_pool=pool,                                         │
│         role=role                                                   │
│     )                                                               │
│                                                                      │
│  3. 创建 Worker 实例：                                               │
│     worker_cls = role_worker_mapping[role]                          │
│     workers = pool.create_workers(worker_cls)                       │
│                                                                      │
│  4. 初始化 Worker：                                                  │
│     for worker in workers:                                          │
│         ray.get(worker.init_model.remote(config))                   │
│                                                                      │
│  结果：                                                              │
│  ─────                                                              │
│  - 每个 Role 对应一个 WorkerGroup                                   │
│  - WorkerGroup 包含多个 Worker 实例                                 │
│  - Worker 已加载模型并准备就绪                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**trainer.fit() 流程：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  trainer.fit() 流程                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PPO 训练循环：                                                      │
│  ─────────                                                          │
│                                                                      │
│  for epoch in range(total_epochs):                                  │
│      for batch in dataloader:                                       │
│          │                                                           │
│          │  Step 1: Rollout (生成 response)                         │
│          │  ───────────────────────────────                         │
│          │  prompts = batch["prompt"]                               │
│          │  responses = actor_rollout_wg.generate(prompts)          │
│          │                                                           │
│          │  Step 2: Reward 计算                                      │
│          │  ─────────────────                                         │
│          │  rewards = reward_fn(prompts + responses)                │
│          │                                                           │
│          │  Step 3: Advantage 计算                                   │
│          │  ─────────────────                                        │
│          │  values = critic_wg.compute_value(prompts + responses)   │
│          │  advantages = compute_gae(rewards, values)               │
│          │                                                           │
│          │  Step 4: Actor 更新                                       │
│          │  ──────────────                                           │
│          │  actor_rollout_wg.update_actor(prompts, responses, adv)  │
│          │                                                           │
│          │  Step 5: Critic 更新                                      │
│          │  ──────────────                                           │
│          │  critic_wg.update_critic(prompts, responses, rewards)    │
│          │                                                           │
│          │  Step 6: 验证 (可选)                                       │
│          │  ──────────────                                           │
│          │  if validation_step:                                      │
│          │      validate()                                           │
│          │                                                           │
│          │  Step 7: 保存 checkpoint (可选)                           │
│          │  ──────────────────────────                               │
│          │  if save_step:                                            │
│          │      save_checkpoint()                                    │
│          │                                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、完整执行流程总结

```
run(config)
│
│  阶段1: 配置初始化
│  ─────────────────────────────────────────────────────────────
│  print(hostname, PID)
│  pprint(config)
│  OmegaConf.resolve(config)
│
│  阶段2: Worker 添加
│  ─────────────────────────────────────────────────────────────
│  actor_rollout_cls = add_actor_rollout_worker(config)
│  add_critic_worker(config)
│  add_reward_model_worker(config)
│  add_ref_policy_worker(config, actor_rollout_cls)
│
│  阶段3: 配置验证
│  ─────────────────────────────────────────────────────────────
│  validate_config(config, need_ref_policy, need_critic)
│
│  阶段4: 模型加载
│  ─────────────────────────────────────────────────────────────
│  local_path = copy_to_local(model.path)
│  tokenizer = hf_tokenizer(local_path)
│  processor = hf_processor(local_path)
│
│  阶段5: Reward Manager
│  ─────────────────────────────────────────────────────────────
│  reward_fn = load_reward_manager(config, tokenizer)
│  val_reward_fn = load_reward_manager(config, tokenizer)
│
│  阶段6: 资源池和数据集
│  ─────────────────────────────────────────────────────────────
│  resource_pool_manager = init_resource_pool_mgr(config)
│  train_dataset = create_rl_dataset(train_files)
│  val_dataset = create_rl_dataset(val_files)
│  train_sampler = create_rl_sampler(config)
│
│  阶段7: Trainer 创建和训练
│  ─────────────────────────────────────────────────────────────
│  trainer = RayPPOTrainer(...)
│  trainer.init_workers()
│  trainer.fit()  # ← 开始 PPO 训练循环
│
```

---

## 五、关键对象生命周期

```
┌─────────────────────────────────────────────────────────────────────┐
│                  关键对象生命周期                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  TaskRunner                                                    │  │
│  │                                                               │  │
│  │  创建时机: run_ppo() 中作为 Ray Actor                          │  │
│  │  生命周期: 单次训练任务                                         │  │
│  │  销毁时机: 训练完成后自动清理                                   │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │ role_worker_mapping (TaskRunner 属性)                   │ │  │
│  │  │                                                         │ │  │
│  │  │ 创建: add_*_worker() 函数逐步填充                       │ │  │
│  │  │ 使用: RayPPOTrainer 初始化时传入                        │ │  │
│  │  │ 生命周期: 传递给 Trainer 后不再使用                     │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  ResourcePoolManager                                           │  │
│  │                                                               │  │
│  │  创建时机: init_resource_pool_mgr()                           │  │
│  │  生命周期: Trainer 初始化 → 训练过程                          │  │
│  │  销毁时机: Trainer 析构                                        │  │
│  │                                                               │  │
│  │  职责: 管理 GPU 资源池，分配 Worker                           │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  RayPPOTrainer                                                 │  │
│  │                                                               │  │
│  │  创建时机: run() 最后阶段                                      │  │
│  │  生命周期: 整个训练过程                                        │  │
│  │  关键方法:                                                     │  │
│  │    - init_workers(): 创建并初始化 Worker                      │  │
│  │    - fit(): 执行 PPO 训练循环                                 │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │ WorkerGroups (Trainer 属性)                             │ │  │
│  │  │                                                         │ │  │
│  │  │ actor_rollout_wg: ActorRollout WorkerGroup             │ │  │
│  │  │ critic_wg: Critic WorkerGroup                          │ │  │
│  │  │ reward_model_wg: RewardModel WorkerGroup (可选)        │ │  │
│  │  │ ref_policy_wg: RefPolicy WorkerGroup (可选)            │ │  │
│  │  │                                                         │ │  │
│  │  │ 每个 WorkerGroup 包含多个 Worker 实例                  │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、总结

| 阶段 | 核心操作 | 关键输出 |
|------|----------|----------|
| 配置初始化 | resolve config | 解析后的配置对象 |
| Worker添加 | add_*_worker() | role_worker_mapping |
| 配置验证 | validate_config() | 配置有效性确认 |
| 模型加载 | copy_to_local + tokenizer | Tokenizer, Processor |
| Reward Manager | load_reward_manager() | reward_fn, val_reward_fn |
| 资源池和数据集 | init_pool + create_dataset | ResourcePoolManager, Datasets |
| Trainer | RayPPOTrainer + fit() | 训练执行 |

**一句话总结：**
`run()` 方法是 PPO 训练的主入口，依次完成配置解析、Worker注册、模型加载、数据准备、Trainer初始化，最后调用 `trainer.fit()` 开始训练循环。

**下一步走读建议：**
继续走读 `RayPPOTrainer.fit()` 方法，了解 PPO 训练循环的详细实现。