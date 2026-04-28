# verl 技术概念速查

> 常见技术概念和配置项的简要说明

---

## 一、资源管理相关

### 1.1 RayResourcePool

**定义**：GPU 资源池管理类，将物理 GPU 抽象为 PlacementGroup 供 WorkerGroup 使用。

**核心功能**：
- 管理多个节点的 GPU 资源
- 创建 Ray PlacementGroup（资源调度单位）
- 支持 GPU 复用（max_colocate_count）
- 支持资源分割和合并

**关键属性**：
| 属性 | 含义 |
|------|------|
| `process_on_nodes` | 每个节点的进程数，如 `[8, 8]` |
| `use_gpu` | 是否分配 GPU |
| `max_colocate_count` | GPU 复用次数，一个 GPU 可分给多个 Worker |

**使用示例**：
```python
resource_pool = RayResourcePool(
    process_on_nodes=[8, 8],  # 两个节点各8进程
    use_gpu=True,
    max_colocate_count=5
)
pgs = resource_pool.get_placement_groups(strategy="STRICT_PACK")
```

---

### 1.2 PlacementGroup 策略

| 策略 | 含义 |
|------|------|
| `STRICT_PACK` | 强制打包，所有 bundle 在同一节点 |
| `PACK` | 尽量打包，优先填满一个节点 |
| `SPREAD` | 尽量分散在不同节点 |
| `STRICT_SPREAD` | 强制分散，每个 bundle 在不同节点 |

---

## 二、Agent 相关

### 2.1 AgentLoopManager

**定义**：多轮 Agent 交互的生成管理器，用于 LLM 与工具/环境的多轮交互场景。

**应用场景**：
- 代码生成 + 执行验证
- Web Agent（浏览器操作）
- 数据分析（SQL 执行）
- 数学推理（中间验证）

**核心架构**：
```
AgentLoopManager
    ├── LLM Servers (多个 vLLM/OpenAI server)
    ├── AgentLoopWorkers (管理单个样本的 agent loop)
    └── RewardModelManager (可选，异步 reward)
```

**关键概念 - response_mask**：
- `1`：LLM 生成的 token（参与梯度计算）
- `0`：工具返回的 token（不参与梯度计算）

---

## 三、验证相关

### 3.1 val_before_train

**作用**：训练前先验证，记录初始模型性能基线。

**用途**：
- 记录初始 accuracy、reward 等指标
- 对比训练后的提升效果
- 检查 reward function 是否正常
- `val_only=True` 时只验证不训练

---

### 3.2 repeat

**作用**：将每个 prompt 复制 n 次，用于生成多个 response。

**验证场景用途**：
- 计算 mean@N：N 个 response 的平均 reward
- 计算 best@N：N 个 response 中最好的 reward
- 计算 pass@N：至少有一个正确的概率

**训练场景用途**：
- GRPO 需要同一 prompt 的多个 response 计算 group advantage

**参数**：
```python
batch = batch.repeat(n=4, interleave=True)
# interleave=True: [p1, p1, p1, p1, p2, p2, p2, p2, ...]
# interleave=False: [p1, p2, p3, p1, p2, p3, ...]
```

---

## 四、数据结构相关

### 4.1 DataProto 结构

```
DataProto
    ├── batch (tensor): PyTorch Tensor
    │   ├── input_ids: [bsz, seq_len]
    │   ├── attention_mask: [bsz, seq_len]
    │   ├── prompts: [bsz, prompt_len]
    │   └── responses: [bsz, response_len]
    │
    ├── non_tensor_batch: numpy array 或 Python 对象
    │   ├── uid: 样本唯一标识
    │   ├── raw_prompt: 原始文本
    │   ├── data_source: 数据来源
    │   ├── reward_model: reward 配置
    │   └── ground_truth: 正确答案
    │
    └── meta_info: 元信息
        ├── eos_token_id
        ├── pad_token_id
        └── global_steps
```

---

### 4.2 non_tensor_batch

**定义**：存放不需要梯度的元信息。

**特点**：
- 不需要 GPU 计算
- 用于统计、过滤、追踪
- 包含文本、ID、配置等

**常见字段**：
| 字段 | 含义 |
|------|------|
| `uid` | 样本唯一标识 |
| `raw_prompt` | 原始文本 prompt |
| `data_source` | 数据来源（math、code 等） |
| `reward_model` | reward 计算配置 |
| `ground_truth` | 正确答案 |

---

### 4.3 uid

**作用**：样本唯一标识符。

**用途**：
- **分组标识**：同一 prompt 的多个 response 共享同一 uid，用于动态采样过滤
- **样本追踪**：记录哪个样本的表现
- **去重**：防止同一 prompt 重复训练
- **数据分析**：分析哪些 prompt 更难、哪些被过滤

**动态采样中的使用**：
```python
# 按 uid 分组计算 reward std
for uid, metric_val in zip(batch.non_tensor_batch["uid"], metrics):
    prompt_uid2metric_vals[uid].append(metric_val)

# std=0 的 uid 被过滤（全对或全错）
```

---

### 4.4 ground_truths

**定义**：数据集中的正确答案。

**来源**：数据集标注（数学题答案、正确代码、QA 正确回答等）

**用途**：
- 计算 reward：比较 model response 和 ground_truth
- 验证时记录：wandb 可视化显示 input、output、ground_truth、score

**示例**：
```python
ground_truths = [
    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
    for item in batch
]
```

---

## 五、生成相关

### 5.1 _get_gen_batch

**作用**：从完整 batch 中分离出生成需要的部分。

**分离原因**：
- `generate_sequences()` 只需要 prompt
- `compute_reward()` 需要 ground_truth + response
- `_get_gen_batch()` 实现分离与后续合并

**保留的 key**：
- `data_source`、`reward_model`、`extra_info`、`uid`
- 这些 key 保留在原 batch，用于后续 reward 计算

**pop 的 key**：
- 所有 tensor batch keys（input_ids, attention_mask 等）
- 其他 non_tensor_batch keys（raw_prompt, ability 等）

**流程**：
```
batch → _get_gen_batch() → gen_batch (用于生成)
                      → batch (保留 reward 配置)
                      
gen_batch → generate_sequences() → gen_batch_output

batch + gen_batch_output → union → 用于 compute_reward
```

---

## 六、分布式相关

### 6.1 size_divisor

**定义**：目标除数，确保 batch_size % size_divisor == 0。

**取值**：
- 标准 Rollout：`world_size`（Worker 数量）
- Agent Loop：`agent.num_workers`

**目的**：负载均衡，每个 Worker 处理相同数量的样本。

---

### 6.2 divisible（可整除）

**数学含义**：A % B == 0

**示例**：
- 16 divisible by 8 → 16 % 8 == 0 ✓
- 13 divisible by 8 → 13 % 8 == 5 ✗

---

### 6.3 pad / unpad

**pad 作用**：填充样本使 batch_size 可整除 size_divisor。

**实现**：
```python
# 13 samples, size_divisor=8
# 13 % 8 = 5 ≠ 0 → 需要 pad 到 16
pad_size = 8 - 5 = 3
# 复制前 3 个样本作为 padding
padded_batch = [s0...s12, s0, s1, s2]  # 16 samples
```

**unpad 作用**：处理完成后丢弃 padding 样本的结果。

```python
output = padded_output[:-pad_size]  # 只保留真实样本结果
```

**完整流程**：
```
原始 batch → pad → 分发给 Workers → 并行处理 → 合并结果 → unpad
```

---

## 七、配置项速查

### 7.1 trust_remote_code

**作用**：控制 HuggingFace 是否执行远程自定义代码。

| 值 | 行为 |
|---|------|
| `False`（默认） | 拒绝执行，报错退出 |
| `True` | 允许执行（需信任模型来源） |

**使用场景**：加载 Qwen、ChatGLM 等附带自定义代码的模型。

---

### 7.2 Nsight

**定义**：NVIDIA GPU 性能分析工具（Nsight Systems）。

**用途**：分析 GPU kernel 执行时间、GPU 利用率、分布式训练瓶颈。

**Ray 配置**：
```python
runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
```

---

## 八、Ray 基础概念

### 8.1 Actor 和 ActorHandle

**Actor**：Ray 远程执行的独立 Python 进程。

**ActorHandle**：Actor 的远程引用，知道 Actor 在哪个进程，可以 RPC 调用。

**创建和使用**：
```python
@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        ...

runner = TaskRunner.remote()  # 创建 Actor，返回 ActorHandle
result = ray.get(runner.run.remote(config))  # RPC 调用 + 等待结果
```

---

### 8.2 .remote() 和 ray.get()

| 操作 | 作用 |
|------|------|
| `Class.remote()` | 创建 Actor，返回 ActorHandle |
| `actor.method.remote(args)` | 异步 RPC 调用，返回 ObjectRef |
| `ray.get(object_ref)` | 阻塞等待，获取执行结果 |

**异步特性**：
```python
# 可以并行发起多个调用
ref1 = runner.run.remote(config1)
ref2 = runner.run.remote(config2)
# 然后一起等待
ray.get([ref1, ref2])
```

---

## 九、算法概念

### 9.1 DAPO 四大创新

| 创新 | 配置 | 含义 |
|------|------|------|
| 不对称裁剪 | `clip_ratio_low/clip_ratio_high` | 正负样本不同裁剪力度 |
| 动态采样 | `filter_groups.enable` | 过滤全对/全错的样本 |
| Token-level Loss | `loss_agg_mode: token-mean` | 按 token 而非 sequence 聚合 |
| 超长惩罚 | `overlong_buffer` | 惩罚过长回答 |

---

### 9.2 动态采样过滤逻辑

**核心**：计算同一 prompt 的 n 个 response 的 reward std。

| std | 判断 | 处理 |
|------|------|------|
| `std = 0` | 全对或全错 | 过滤，重新采样 |
| `std > 0` | 有差异 | 保留，用于训练 |

**原因**：GRPO advantage 计算 `A = (r - mean) / std`，std=0 时无法计算。

---

## 十、总结表

| 类别 | 关键概念 |
|------|----------|
| **资源管理** | RayResourcePool, PlacementGroup, max_colocate_count |
| **Agent** | AgentLoopManager, response_mask, 多轮交互 |
| **验证** | val_before_train, repeat, mean@N/best@N |
| **数据结构** | DataProto, non_tensor_batch, uid, ground_truth |
| **生成** | _get_gen_batch, 分离 prompt 和 reward 配置 |
| **分布式** | size_divisor, divisible, pad/unpad, 负载均衡 |
| **配置** | trust_remote_code, Nsight profiling |
| **Ray 基础** | Actor, ActorHandle, .remote(), ray.get() |
| **算法** | DAPO 四大创新，动态采样，GRPO advantage |