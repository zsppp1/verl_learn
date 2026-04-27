# create_rl_dataset 与 create_rl_sampler 函数解析

> verl/trainer/main_ppo.py 第365-439行

## 一、create_rl_dataset 函数

### 1.1 函数定义

```python
def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
```

**核心职责：**
- 根据配置获取合适的 Dataset 类
- 创建 Dataset 实例
- 返回数据集供训练使用

---

### 1.2 执行流程

```python
from verl.utils.dataset.rl_dataset import get_dataset_class

# Get the dataset class
dataset_cls = get_dataset_class(data_config)

# Instantiate the dataset using the determined dataset class
dataset = dataset_cls(
    data_files=data_paths,
    tokenizer=tokenizer,
    processor=processor,
    config=data_config,
    max_samples=max_samples,
)

return dataset
```

**流程图：**

```
create_rl_dataset(data_paths, data_config, tokenizer, processor)
│
│  选择 Dataset 类
│  ─────────────────────────────────────────────────────────────
│  dataset_cls = get_dataset_class(data_config)
│  
│  根据 data_config.dataset_type 或其他配置选择：
│  - JsonlDataset (最常见)
│  - ParquetDataset
│  - CustomDataset
│
│  创建 Dataset 实例
│  ─────────────────────────────────────────────────────────────
│  dataset = dataset_cls(
│      data_files=data_paths,    # 数据文件路径列表
│      tokenizer=tokenizer,      # Tokenizer 实例
│      processor=processor,      # Processor 实例（多模态）
│      config=data_config,       # 数据配置
│      max_samples=max_samples   # 最大样本数
│  )
│
│  返回
│  ─────────────────────────────────────────────────────────────
│  return dataset
│
```

---

### 1.3 Dataset 类详解

**常见 Dataset 类型：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  RL Dataset 类型                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  JsonlDataset:                                                       │
│  ────────────                                                        │
│  - 最常用的数据格式                                                   │
│  - 支持 JSON Lines 格式                                              │
│  - 每行一个 JSON 对象                                                 │
│                                                                      │
│  数据格式示例：                                                       │
│  {"prompt": "请解释什么是PPO", "responses": ["PPO是..."]}            │
│  {"prompt": "写一个排序算法", "responses": ["def sort..."]}          │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  ParquetDataset:                                                     │
│  ──────────────                                                      │
│  - 高效的列存储格式                                                   │
│  - 适合大规模数据集                                                   │
│  - 更快的读取速度                                                    │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  CustomDataset:                                                      │
│  ──────────────                                                      │
│  - 用户自定义数据集                                                   │
│  - 通过配置指定类路径                                                 │
│                                                                      │
│  配置示例：                                                           │
│  dataset:                                                            │
│    class_path: "my_module.dataset"                                   │
│    class_name: "MyCustomDataset"                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 1.4 max_samples 参数

```python
max_samples: int = -1
```

**用途：**

| 值 | 意义 | 使用场景 |
|----|------|----------|
| `-1` (默认) | 使用全部数据 | 正常训练 |
| `N` (> 0) | 只使用前 N 个样本 | 快速调试、测试 |

**调试场景：**

```python
# 快速测试训练流程
train_dataset = create_rl_dataset(
    ..., 
    max_samples=100  # 只用 100 个样本快速验证
)
```

---

### 1.5 Dataset 初始化过程

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Dataset 初始化过程                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  dataset_cls.__init__() 执行：                                       │
│  ─────────────────────────────                                       │
│                                                                      │
│  1. 加载数据文件：                                                   │
│     with open(data_file, 'r') as f:                                 │
│         self.data = [json.loads(line) for line in f]                │
│                                                                      │
│  2. Tokenize prompts:                                               │
│     for item in self.data:                                          │
│         item['prompt_tokens'] = tokenizer.encode(item['prompt'])    │
│                                                                      │
│  3. 处理多模态数据 (如果 processor 存在):                             │
│     if processor:                                                   │
│         item['images'] = processor.load_images(item['images'])      │
│                                                                      │
│  4. 应用 max_samples:                                                │
│     if max_samples > 0:                                             │
│         self.data = self.data[:max_samples]                         │
│                                                                      │
│  结果：                                                              │
│  ─────                                                              │
│  - self.data: 处理后的数据列表                                       │
│  - self.tokenizer: Tokenizer 引用                                   │
│  - self.processor: Processor 引用（可选）                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、create_rl_sampler 函数

### 2.1 函数定义

```python
def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
```

**核心职责：**
- 根据配置选择采样策略
- 支持课程学习、随机采样、顺序采样
- 返回 Sampler 实例

---

### 2.2 三种采样策略

```
                ┌─────────────────────────────────┐
                │   sampler.class_path = ?        │
                └─────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
            有配置                        无配置
              │                             │
              ▼                             ▼
    ┌──────────────────┐          ┌─────────────────────┐
    │ 课程学习 Sampler │          │ 检查 shuffle = ?    │
    │ (Curriculum)     │          └─────────────────────┘
    └──────────────────┘                    │
                                  ┌────────┼────────┐
                                  │                 │
                                True             False
                                  │                 │
                                  ▼                 ▼
                          ┌────────────┐    ┌──────────────┐
                          │ Random     │    │ Sequential   │
                          │ Sampler    │    │ Sampler      │
                          │ (随机采样) │    │ (顺序采样)   │
                          └────────────┘    └──────────────┘
```

---

### 2.3 课程学习 Sampler

```python
if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
    curriculum_class = load_extern_object(
        data_config.sampler.class_path,
        data_config.sampler.class_name,
    )
    sampler = curriculum_class(
        data_source=dataset,
        data_config=data_config,
    )
    assert isinstance(sampler, AbstractSampler)
    assert data_config.get("dataloader_num_workers", 8) == 0, (
        "If using curriculum, num_workers must be 0 to prevent data caching. "
        "If the dataloader caches data before the batch is done the "
        "curriculum sampler won't have the opportunity to reorder it. "
    )
```

**课程学习概念：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  课程学习 (Curriculum Learning)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  核心思想：                                                          │
│  ─────────                                                          │
│  - 从简单样本开始训练                                                │
│  - 逐渐增加样本难度                                                  │
│  - 类似人类学习过程                                                  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   训练阶段 1: 简单样本                                        │  │
│  │   ─────────────────                                          │  │
│  │   难度 = 1-3                                                  │  │
│  │   例如：短文本、简单问题                                      │  │
│  │                                                               │  │
│  │   训练阶段 2: 中等样本                                        │  │
│  │   ─────────────────                                          │  │
│  │   难度 = 4-6                                                  │  │
│  │                                                               │  │
│  │   训练阶段 3: 困难样本                                        │  │
│  │   ─────────────────                                          │  │
│  │   难度 = 7-10                                                 │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Sampler 的职责：                                                    │
│  ─────────────                                                      │
│  - 决定样本的采样顺序                                                │
│  - 根据当前训练进度动态调整                                          │
│  - 可能根据模型性能反馈重新排序                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**num_workers = 0 的原因：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  为什么课程学习需要 num_workers = 0                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DataLoader 的数据缓存行为：                                         │
│  ──────────────────────────                                         │
│                                                                      │
│  num_workers > 0:                                                   │
│  ─────────────────                                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  Worker 进程会预取多个 batch                                  │  │
│  │                                                               │  │
│  │  Main Process                  Worker Processes              │  │
│  │  ──────────────                ───────────────────           │  │
│  │                                                               │  │
│  │                                [Batch 1] ← 已缓存             │  │
│  │                                [Batch 2] ← 已缓存             │  │
│  │                                [Batch 3] ← 已缓存             │  │
│  │                                [Batch 4] ← 已缓存             │  │
│  │                                                               │  │
│  │  问题：                                                       │  │
│  │  课程学习 Sampler 无法动态调整已缓存的 batch                   │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  num_workers = 0:                                                   │
│  ─────────────────                                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  单进程，每次调用 Sampler 获取下一个 batch                    │  │
│  │                                                               │  │
│  │  Sampler 可以实时调整采样顺序                                 │  │
│  │                                                               │  │
│  │  Curriculum Sampler                                           │  │
│  │  ─────────────────                                            │  │
│  │  每次调用 __iter__() 都可以重新排序                           │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  源码注释原文：                                                      │
│  ─────────────────                                                   │
│  "If the dataloader caches data before the batch is done the       │
│   curriculum sampler won't have the opportunity to reorder it."     │
│                                                                      │
│  翻译：                                                              │
│  "如果 dataloader 在 batch 完成前就缓存了数据，                      │
│   课程学习 sampler 就没有机会重新排序了。"                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.4 Random Sampler

```python
# torch.utils.data.RandomSampler could not recover properly
from torchdata.stateful_dataloader.sampler import RandomSampler

elif data_config.shuffle:
    train_dataloader_generator = torch.Generator()
    seed = data_config.get("seed")
    if seed is not None:
        train_dataloader_generator.manual_seed(seed)
    sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
```

**为什么使用 stateful_dataloader 的 RandomSampler？**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Stateful RandomSampler vs 普通 RandomSampler        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  普通 torch.utils.data.RandomSampler:                                │
│  ───────────────────────────────────────────                         │
│  ❌ 无法正确恢复状态                                                 │
│  ❌ checkpoint 恢复后采样顺序会改变                                  │
│  ❌ 影响训练可复现性                                                 │
│                                                                      │
│  torchdata.stateful_dataloader.sampler.RandomSampler:               │
│  ───────────────────────────────────────────────────────────         │
│  ✅ 支持状态保存和恢复                                               │
│  ✅ checkpoint 恢复后采样顺序一致                                    │
│  ✅ 训练可复现                                                      │
│                                                                      │
│  Generator + seed:                                                   │
│  ─────────────────                                                   │
│  - 使用 torch.Generator 控制随机性                                  │
│  - seed 固定时，采样顺序完全可复现                                   │
│  - checkpoint 恢复时恢复 Generator 状态                             │
│                                                                      │
│  示例：                                                              │
│  ─────                                                              │
│                                                                      │
│  seed = 42                                                          │
│  generator.manual_seed(seed)                                        │
│                                                                      │
│  epoch 1: [3, 7, 1, 5, 2, ...]  ← 随机顺序                         │
│  checkpoint saved                                                   │
│                                                                      │
│  epoch 2 (恢复后): [4, 8, 6, 9, ...]  ← 继续之前的状态              │
│                                                                      │
│  而不是从头重新随机：                                                │
│  epoch 2 (普通 sampler): [3, 7, 1, 5, 2, ...]  ← 重新开始           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.5 Sequential Sampler

```python
else:
    # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
    sampler = SequentialSampler(data_source=dataset)
```

**顺序采样特点：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Sequential Sampler                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  特点：                                                              │
│  ─────                                                              │
│  - 按数据集顺序逐个采样                                              │
│  - 不随机，确定性顺序                                                │
│  - 适合验证集或需要顺序处理的场景                                    │
│                                                                      │
│  采样顺序：                                                          │
│  ─────────                                                          │
│  [Sample 0, Sample 1, Sample 2, Sample 3, ...]                      │
│                                                                      │
│  使用场景：                                                          │
│  ─────────                                                          │
│  - shuffle = False                                                  │
│  - 验证集评估（顺序一致便于对比）                                    │
│  - 数据有顺序依赖时                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.6 源码注释解读

```python
# Use a sampler to facilitate checkpoint resumption.
```

**翻译：**
> "使用 sampler 来便于 checkpoint 恢复"

**checkpoint 恢复流程：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Sampler 在 checkpoint 恢复中的作用                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  checkpoint 保存内容：                                               │
│  ─────────────────────                                              │
│  1. 模型参数                                                         │
│  2. Optimizer 状态                                                   │
│  3. Sampler 状态 ← 重要！                                           │
│  4. 当前 epoch / step                                               │
│                                                                      │
│  恢复流程：                                                          │
│  ─────────                                                          │
│                                                                      │
│  load_checkpoint()                                                  │
│  │                                                                   │
│  │  恢复模型参数                                                     │
│  │  恢复 Optimizer                                                   │
│  │  恢复 Sampler 状态 ← 确保数据顺序一致                             │
│  │  恢复 epoch/step                                                  │
│  │                                                                   │
│  │  continue training from saved state                              │
│  │                                                                   │
│  结果：                                                              │
│  ─────                                                              │
│  训练从 checkpoint 处继续，                                          │
│  数据采样顺序与保存时一致                                            │
│                                                                      │
│  没有 Sampler 状态恢复：                                             │
│  ─────────────────────                                               │
│  - 数据从头开始采样                                                  │
│  - 可能重复训练已训练过的数据                                        │
│  - 或者错过未训练的数据                                              │
│  - 训练效果不一致                                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、Sampler 类型对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Sampler 类型对比表                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┬──────────────┬──────────────┬───────────────────┐  │
│  │ 类型        │ 采样顺序     │ 状态恢复     │ 使用场景          │  │
│  ├─────────────┼──────────────┼──────────────┼───────────────────┤  │
│  │ Curriculum  │ 动态调整     │ ✅ 支持      │ 渐进式学习        │  │
│  │ Sampler     │ (按难度)     │              │                   │  │
│  ├─────────────┼──────────────┼──────────────┼───────────────────┤  │
│  │ Random      │ 随机顺序     │ ✅ 支持      │ 标准训练          │  │
│  │ Sampler     │ (seed控制)   │              │ (shuffle=True)    │  │
│  ├─────────────┼──────────────┼──────────────┼───────────────────┤  │
│  │ Sequential  │ 固定顺序     │ ✅ 支持      │ 验证集            │  │
│  │ Sampler     │ (按索引)     │              │ (shuffle=False)   │  │
│  └─────────────┴──────────────┴──────────────┴───────────────────┘  │
│                                                                      │
│  选择逻辑：                                                          │
│  ─────────                                                          │
│                                                                      │
│  1. sampler.class_path 配置存在 → Curriculum Sampler               │
│  2. shuffle = True → Random Sampler                                 │
│  3. shuffle = False → Sequential Sampler                            │
│                                                                      │
│  配置示例：                                                          │
│  ─────────                                                          │
│                                                                      │
│  # 课程学习                                                          │
│  data:                                                               │
│    sampler:                                                          │
│      class_path: "verl.experimental.curriculum"                     │
│      class_name: "DifficultySampler"                                │
│    dataloader_num_workers: 0  # 必须为 0                            │
│                                                                      │
│  # 随机采样                                                          │
│  data:                                                               │
│    shuffle: true                                                    │
│    seed: 42                                                         │
│                                                                      │
│  # 顺序采样                                                          │
│  data:                                                               │
│    shuffle: false                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、调用时机

在 `TaskRunner.run()` 中调用：

```python
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

**注意：**
- 训练集和验证集使用不同的数据文件
- 训练集和验证集可以有不同的 `max_samples` 限制
- Sampler 通常只用于训练集，验证集使用顺序采样

---

## 五、数据流转完整流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                  数据流转完整流程                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 数据文件                                                         │
│  ─────────                                                          │
│     train.jsonl / val.jsonl                                         │
│                                                                      │
│  2. Dataset 创建                                                     │
│  ─────────                                                          │
│     create_rl_dataset()                                             │
│     │                                                                │
│     │  加载文件 → tokenize → 处理多模态 → max_samples               │
│     │                                                                │
│     ↓                                                                │
│     Dataset 实例                                                     │
│                                                                      │
│  3. Sampler 创建                                                     │
│  ─────────                                                          │
│     create_rl_sampler()                                             │
│     │                                                                │
│     │  根据配置选择：Curriculum / Random / Sequential               │
│     │                                                                │
│     ↓                                                                │
│     Sampler 实例                                                     │
│                                                                      │
│  4. DataLoader 创建 (在 RayPPOTrainer)                              │
│  ───────────────────────────────────────                            │
│     DataLoader(                                                     │
│         dataset=train_dataset,                                      │
│         sampler=train_sampler,                                      │
│         batch_size=...,                                             │
│         collate_fn=...,                                             │
│         num_workers=...                                             │
│     )                                                               │
│                                                                      │
│  5. 训练循环                                                         │
│  ─────────                                                          │
│     for batch in dataloader:                                        │
│         # PPO 训练步骤                                               │
│         prompts = batch['prompts']                                  │
│         ...                                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 六、总结

### create_rl_dataset

| 层级 | 内容 |
|------|------|
| **输入** | data_paths, data_config, tokenizer, processor, max_samples |
| **核心操作** | get_dataset_class → dataset_cls → dataset |
| **输出** | Dataset 实例 |

**一句话总结：**
根据配置获取合适的 Dataset 类，创建数据集实例并完成 tokenize 和预处理。

### create_rl_sampler

| 层级 | 内容 |
|------|------|
| **输入** | data_config, dataset |
| **决策** | sampler.class_path → shuffle → Sequential |
| **输出** | Sampler 实例 |

**一句话总结：**
根据配置选择采样策略（课程学习、随机、顺序），创建支持 checkpoint 恢复的 Sampler 实例。