# ResourcePoolManager 类详解

> verl/trainer/ppo/ray_trainer.py 行70-124

## 一、类定义

```python
@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
```

**核心职责：**
- 定义和管理 GPU 资源池规格
- 创建 RayResourcePool 实例
- 提供按 Role 获取资源池的接口
- 检查资源可用性

---

## 二、属性详解

```python
resource_pool_spec: dict[str, list[int]]
mapping: dict[Role, str]
resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
```

**属性说明：**

| 属性 | 类型 | 用途 |
|------|------|------|
| `resource_pool_spec` | `dict[str, list[int]]` | 资源池规格定义 |
| `mapping` | `dict[Role, str]` | Role → Pool ID 映射 |
| `resource_pool_dict` | `dict[str, RayResourcePool]` | 实际创建的资源池 |

**数据结构示例：**

```python
# resource_pool_spec
{
    "global_pool": [8, 8, 8],   # 3节点，每节点8GPU
    "reward_pool": [4, 4]       # 2节点，每节点4GPU
}

# mapping (来自 main_ppo.py)
{
    Role.ActorRollout: "global_pool",
    Role.Critic: "global_pool",
    Role.RewardModel: "reward_pool",
    Role.RefPolicy: "global_pool"
}

# resource_pool_dict (创建后)
{
    "global_pool": RayResourcePool(...),
    "reward_pool": RayResourcePool(...)
}
```

---

## 三、create_resource_pool 方法

```python
def create_resource_pool(self):
    """Create Ray resource pools for distributed training.

    Initializes resource pools based on the resource pool specification,
    with each pool managing GPU resources across multiple nodes.
    For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
    For Megatron backend, uses max_colocate_count>1 for different models.
    """
    for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
        # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
        # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
        # For Megatron backend, we recommend using max_colocate_count>1
        # that can utilize different WorkerGroup for differnt models
        resource_pool = RayResourcePool(
            process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
        )
        self.resource_pool_dict[resource_pool_name] = resource_pool

    self._check_resource_available()
```

**max_colocate_count 参数详解：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  max_colocate_count 参数                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  源码注释解读：                                                      │
│  ─────────────                                                      │
│                                                                      │
│  "max_colocate_count means the number of WorkerGroups               │
│   (i.e. processes) in each RayResourcePool"                         │
│                                                                      │
│  翻译：                                                              │
│  "max_colocate_count 表示每个 RayResourcePool 中                    │
│   WorkerGroup（即进程）的数量"                                       │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  FSDP backend (max_colocate_count=3):                               │
│  ───────────────────────────────────────                            │
│                                                                      │
│  一个 RayResourcePool 可容纳最多 3 个 WorkerGroup：                 │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    global_pool                                 │  │
│  │                                                               │  │
│  │   WorkerGroup 1: Actor + Critic + RefPolicy                   │  │
│  │   WorkerGroup 2: Rollout                                      │  │
│  │   WorkerGroup 3: Reward Model (optional)                      │  │
│  │                                                               │  │
│  │   共享同一 GPU 资源                                            │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  优点：                                                              │
│  ✅ 多个 WorkerGroup 共享 GPU，减少显存占用                         │
│  ✅ Actor/Critic/RefPolicy 可以在同一 GPU 上                       │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  Megatron backend (max_colocate_count>1):                           │
│  ──────────────────────────────────────────────                     │
│                                                                      │
│  Megatron-LM 的分布式架构需要更多独立进程：                         │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   不同模型类型使用不同的 WorkerGroup                          │  │
│  │                                                               │  │
│  │   WorkerGroup 1: Actor (Megatron TP+PP)                      │  │
│  │   WorkerGroup 2: Critic (Megatron TP+PP)                     │  │
│  │   WorkerGroup 3: ...                                         │  │
│  │                                                               │  │
│  │   每个 WorkerGroup 有独立的分布式配置                         │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、get_resource_pool 方法

```python
def get_resource_pool(self, role: Role) -> RayResourcePool:
    """Get the resource pool of the worker_cls"""
    return self.resource_pool_dict[self.mapping[role]]
```

**使用示例：**

```python
# 获取 Actor 对应的资源池
actor_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)

# 内部执行：
# 1. self.mapping[Role.ActorRollout] = "global_pool"
# 2. self.resource_pool_dict["global_pool"] = RayResourcePool(...)
# 3. 返回 RayResourcePool 实例
```

---

## 五、get_n_gpus 方法

```python
def get_n_gpus(self) -> int:
    """Get the number of gpus in this cluster."""
    return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
```

**计算逻辑：**

```python
# resource_pool_spec = {"global_pool": [8, 8, 8], "reward_pool": [4, 4]}

# 计算：
# global_pool: 8 + 8 + 8 = 24
# reward_pool: 4 + 4 = 8
# 总计: 24 + 8 = 32 GPU

return 32
```

---

## 六、_check_resource_available 方法

```python
def _check_resource_available(self):
    """Check if the resource pool can be satisfied in this ray cluster."""
    node_available_resources = ray._private.state.available_resources_per_node()
    node_available_gpus = {
        node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
        for node, node_info in node_available_resources.items()
    }

    # check total required gpus can be satisfied
    total_available_gpus = sum(node_available_gpus.values())
    total_required_gpus = sum(
        [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
    )
    if total_available_gpus < total_required_gpus:
        raise ValueError(
            f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
        )
```

**检查流程：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  _check_resource_available 流程                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: 获取 Ray 集群可用资源                                       │
│  ───────────────────────────────                                    │
│  ray._private.state.available_resources_per_node()                  │
│                                                                      │
│  返回每个节点的可用资源：                                            │
│  {                                                                  │
│      "node-1": {"GPU": 8, "CPU": 64},                              │
│      "node-2": {"GPU": 8, "CPU": 64},                              │
│      "node-3": {"GPU": 8, "CPU": 64}                               │
│  }                                                                  │
│                                                                      │
│  Step 2: 提取 GPU 数量                                               │
│  ───────────────────                                                │
│  node_available_gpus = {"node-1": 8, "node-2": 8, "node-3": 8}      │
│                                                                      │
│  Step 3: 计算总数                                                    │
│  ─────────                                                          │
│  total_available_gpus = 8 + 8 + 8 = 24                             │
│  total_required_gpus = 24 (from resource_pool_spec)                │
│                                                                      │
│  Step 4: 检查是否足够                                                │
│  ─────────────                                                      │
│  if total_available_gpus < total_required_gpus:                    │
│      raise ValueError("GPU 不够")                                  │
│                                                                      │
│  注意：                                                              │
│  ─────                                                              │
│  - 支持 GPU 和 NPU（Ascend AI 处理器）                              │
│  - 检查总数，不检查单个节点                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、与 RayResourcePool 的关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                  ResourcePoolManager vs RayResourcePool              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ResourcePoolManager (管理层):                                      │
│  ───────────────────────────                                        │
│  - 管理多个资源池                                                    │
│  - 提供 Role → Pool 映射                                            │
│  - 检查资源可用性                                                    │
│  - 不直接管理 GPU                                                   │
│                                                                      │
│  RayResourcePool (执行层):                                          │
│  ───────────────────────                                            │
│  - 实际管理 GPU 资源                                                │
│  - 创建和调度 Worker                                                │
│  - 执行分布式任务                                                   │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   ResourcePoolManager                                         │  │
│  │   ┌─────────────────────┐                                    │  │
│  │   │ resource_pool_spec  │                                    │  │
│  │   │ {"global": [8,8,8]} │                                    │  │
│  │   └─────────────────────┘                                    │  │
│  │           │                                                   │  │
│  │           │ create_resource_pool()                           │  │
│  │           ↓                                                   │  │
│  │   ┌─────────────────────────────────────────────────────┐    │  │
│  │   │                  RayResourcePool                     │    │  │
│  │   │                                                     │    │  │
│  │   │   process_on_nodes=[8,8,8]                          │    │  │
│  │   │   use_gpu=True                                      │    │  │
│  │   │   max_colocate_count=3                              │    │  │
│  │   │                                                     │    │  │
│  │   │   内部管理：                                         │    │  │
│  │   │   - GPU 分配                                        │    │  │
│  │   │   - Worker 创建                                     │    │  │
│  │   │   - 任务调度                                        │    │  │
│  │   │                                                     │    │  │
│  │   └─────────────────────────────────────────────────────┘    │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 八、调用时机

在 `RayPPOTrainer.init_workers()` 中调用：

```python
def init_workers(self):
    # Step 1: 创建资源池
    self.resource_pool_manager.create_resource_pool()  # ← 这里调用
    
    # Step 2: 获取资源池并创建 WorkerGroup
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
    
    # Step 3: 在资源池上创建 Worker
    worker_group = RayWorkerGroup(resource_pool=resource_pool, ...)
```

---

## 九、总结

| 方法 | 职责 |
|------|------|
| `create_resource_pool()` | 创建 RayResourcePool 实例 |
| `get_resource_pool(role)` | 按 Role 获取对应资源池 |
| `get_n_gpus()` | 计算总 GPU 数量 |
| `_check_resource_available()` | 检查资源是否足够 |

**一句话总结：**
ResourcePoolManager 是 GPU 资源池的管理层，负责创建、查询和校验 RayResourcePool，为 RayPPOTrainer 提供 Role → ResourcePool 的映射服务。