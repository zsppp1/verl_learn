# init_workers 方法详解

> verl/trainer/ppo/ray_trainer.py 行751-936

## 一、方法定义

```python
def init_workers(self):
    """Initialize distributed training workers using Ray backend.

    Creates:
    1. Ray resource pools from configuration
    2. Worker groups for each role (actor, critic, etc.)
    """
```

**核心职责：**
- 创建 RayResourcePool
- 为每个 Role 创建 WorkerGroup
- 初始化各个 Worker 的模型

---

## 二、整体流程图

```
init_workers()
│
│  Step 1: 创建资源池
│  ─────────────────────────────────────────────────────────────
│  self.resource_pool_manager.create_resource_pool()
│
│  Step 2: 准备 Worker 类字典
│  ─────────────────────────────────────────────────────────────
│  self.resource_pool_to_cls = {pool: {} for pool in pools}
│
│  Step 3: 创建 Actor/Rollout Worker
│  ─────────────────────────────────────────────────────────────
│  actor_role = Role.ActorRolloutRef or Role.ActorRollout
│  actor_rollout_cls = RayClassWithInitArgs(...)
│  resource_pool_to_cls[pool][actor_role] = actor_rollout_cls
│
│  Step 4: 创建 Critic Worker (if use_critic)
│  ─────────────────────────────────────────────────────────────
│  critic_cls = RayClassWithInitArgs(...)
│  resource_pool_to_cls[pool][Role.Critic] = critic_cls
│
│  Step 5: 创建 RefPolicy Worker (if use_ref_policy)
│  ─────────────────────────────────────────────────────────────
│  ref_policy_cls = RayClassWithInitArgs(...)
│  resource_pool_to_cls[pool][Role.RefPolicy] = ref_policy_cls
│
│  Step 6: 创建 RewardModel Worker (if use_rm)
│  ─────────────────────────────────────────────────────────────
│  rm_cls = RayClassWithInitArgs(...)
│  resource_pool_to_cls[pool][Role.RewardModel] = rm_cls
│
│  Step 7: 创建 WorkerGroup
│  ─────────────────────────────────────────────────────────────
│  for resource_pool, class_dict in resource_pool_to_cls.items():
│      worker_dict_cls = create_colocated_worker_cls(class_dict)
│      wg_dict = RayWorkerGroup(resource_pool, worker_dict_cls)
│      spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
│      all_wg.update(spawn_wg)
│
│  Step 8: 初始化各 WorkerGroup 的模型
│  ─────────────────────────────────────────────────────────────
│  critic_wg.init_model() (or reset + set_loss_fn)
│  ref_policy_wg.init_model()
│  rm_wg.init_model()
│  actor_rollout_wg.init_model()
│
│  Step 9: 创建 AsyncRolloutManager
│  ─────────────────────────────────────────────────────────────
│  self.async_rollout_manager = AgentLoopManager(...)
│
```

---

## 三、详细代码解析

### 3.1 Step 1: 创建资源池

```python
self.resource_pool_manager.create_resource_pool()

self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
```

**resource_pool_to_cls 结构：**

```python
# 创建后的结构
{
    RayResourcePool("global_pool"): {},
    RayResourcePool("reward_pool"): {}  # 如果有独立池
}

# 后续会填充每个 pool 对应的 Worker 类
```

---

### 3.2 Step 3: 创建 Actor/Rollout Worker

```python
actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout

if self.hybrid_engine:
    resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[actor_role],
        config=self.config.actor_rollout_ref,
        role=str(actor_role),
    )
    self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
```

**RayClassWithInitArgs 的作用：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  RayClassWithInitArgs 包装                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  RayClassWithInitArgs 是一个包装器：                                 │
│  ─────────────────────────────────                                   │
│                                                                      │
│  将 Worker 类和初始化参数打包在一起                                  │
│                                                                      │
│  RayClassWithInitArgs(                                              │
│      cls=ActorRolloutRefWorker,      # Worker 类                   │
│      config=config,                  # 配置参数                    │
│      role="ActorRolloutRef",         # Role 名称                   │
│  )                                                                  │
│                                                                      │
│  用途：                                                              │
│  ─────                                                              │
│  1. WorkerGroup 创建时，使用这个包装器                              │
│  2. 自动将 config 和 role 传递给 Worker.__init__()                  │
│  3. 避免手动传参数                                                  │
│                                                                      │
│  类比：                                                              │
│  ─────                                                              │
│  RayClassWithInitArgs = Worker + 初始化参数                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.3 Step 4: 创建 Critic Worker

```python
if self.use_critic:
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
    
    from verl.workers.config import CriticConfig
    critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)
    
    # 新版引擎转换配置
    if self.use_legacy_worker_impl == "disable":
        from verl.workers.engine_workers import TrainingWorkerConfig
        
        # convert critic_cfg into TrainingWorkerConfig
        critic_cfg = TrainingWorkerConfig(
            model_type="value_model",
            model_config=orig_critic_cfg.model_config,
            engine_config=engine_config,
            optimizer_config=orig_critic_cfg.optim,
            checkpoint_config=orig_critic_cfg.checkpoint,
        )
    
    critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
    self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls
```

**新版引擎的配置转换：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Critic 配置转换 (新版引擎)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  旧版 CriticConfig → 新版 TrainingWorkerConfig                      │
│  ─────────────────────────────────────────────                      │
│                                                                      │
│  TrainingWorkerConfig(                                              │
│      model_type="value_model",        # 指定是 Value Model         │
│      model_config=...,               # 模型配置                    │
│      engine_config=...,              # 引擎配置                    │
│      optimizer_config=...,           # 优化器配置                  │
│      checkpoint_config=...,          # checkpoint 配置             │
│  )                                                                  │
│                                                                      │
│  原因：                                                              │
│  ─────                                                              │
│  新版引擎使用通用 TrainingWorker                                    │
│  需要通过 model_type 区分 Actor/Critic                             │
│                                                                      │
│  model_type 类型：                                                   │
│  ─────────────                                                      │
│  - "actor": Actor 模型                                              │
│  - "value_model": Critic/Value 模型                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.4 Step 7: 创建 WorkerGroup

```python
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = self.ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls,
        **wg_kwargs,
    )
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    all_wg.update(spawn_wg)
```

**create_colocated_worker_cls 作用：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  create_colocated_worker_cls 详解                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入 class_dict:                                                   │
│  ─────────────                                                      │
│  {                                                                  │
│      "ActorRolloutRef": RayClassWithInitArgs(ActorRolloutRefWorker),│
│      "Critic": RayClassWithInitArgs(CriticWorker),                 │
│  }                                                                  │
│                                                                      │
│  create_colocated_worker_cls(class_dict)                            │
│                                                                      │
│  输出: 一个组合的 Worker 类                                         │
│  ─────────────────────                                              │
│  将多个 Worker 类合并为一个类                                       │
│  同一个进程可以处理多个 Role                                        │
│                                                                      │
│  优点：                                                              │
│  ─────                                                              │
│  ✅ 共享 GPU 显存                                                   │
│  ✅ Actor + Critic 可以在同一 GPU                                  │
│  ✅ 减少进程数量                                                    │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   Colocated Worker                                           │  │
│  │                                                               │  │
│  │   ┌─────────────────────────────────────────────────────┐    │  │
│  │   │                                                     │    │  │
│  │   │   ActorRolloutRefWorker + CriticWorker              │    │  │
│  │   │                                                     │    │  │
│  │   │   一个进程，两个模型                                 │    │  │
│  │   │                                                     │    │  │
│  │   └─────────────────────────────────────────────────────┘    │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  spawn(prefix_set):                                                 │
│  ─────────────                                                      │
│  从 wg_dict 中按 prefix (Role名) 分离出单独的 WorkerGroup          │
│                                                                      │
│  返回:                                                              │
│  {                                                                  │
│      "ActorRolloutRef": WorkerGroup1,                              │
│      "Critic": WorkerGroup2,                                       │
│  }                                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.5 Step 8: 初始化模型

```python
if self.use_critic:
    self.critic_wg = all_wg[str(Role.Critic)]
    if self.use_legacy_worker_impl == "disable":
        self.critic_wg.reset()
        # assign critic loss
        from verl.workers.utils.losses import value_loss
        value_loss_ = partial(value_loss, config=orig_critic_cfg)
        self.critic_wg.set_loss_fn(value_loss_)
    else:
        self.critic_wg.init_model()

if self.use_reference_policy and not self.ref_in_actor:
    if str(Role.RefPolicy) in all_wg:
        self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
        self.ref_policy_wg.init_model()
    else:
        # Model engine: ActorRolloutRefWorker
        self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

self.rm_wg = None
if self.use_rm and not self.use_reward_loop:
    self.rm_wg = all_wg[str(Role.RewardModel)]
    self.rm_wg.init_model()

self.actor_rollout_wg = all_wg[str(actor_role)]
self.actor_rollout_wg.init_model()
```

**初始化顺序的意义：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Worker 初始化顺序                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  顺序：                                                              │
│  ─────                                                              │
│  1. Critic → init_model/reset                                       │
│  2. RefPolicy → init_model                                          │
│  3. RewardModel → init_model                                        │
│  4. ActorRollout → init_model (最后)                               │
│                                                                      │
│  为什么 ActorRollout 最后？                                          │
│  ─────────────────────────                                           │
│                                                                      │
│  源码注释：                                                          │
│  ─────────                                                          │
│  "we should create rollout at the end so that vllm                 │
│   can have a better estimation of kv cache memory"                 │
│                                                                      │
│  翻译：                                                              │
│  "我们应该最后创建 rollout，这样 vLLM 可以更好地估算 kv cache 内存" │
│                                                                      │
│  原因：                                                              │
│  ─────                                                              │
│  vLLM 需要估算 KV Cache 内存                                        │
│  其他模型加载后，剩余内存更明确                                      │
│  vLLM 可以根据剩余内存优化 KV Cache                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.6 新版引擎的 Critic 初始化

```python
if self.use_legacy_worker_impl == "disable":
    self.critic_wg.reset()
    # assign critic loss
    from verl.workers.utils.losses import value_loss
    value_loss_ = partial(value_loss, config=orig_critic_cfg)
    self.critic_wg.set_loss_fn(value_loss_)
```

**新版引擎的 loss 设置：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  新版引擎 Critic Loss 设置                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  旧版 CriticWorker:                                                 │
│  ─────────────────                                                  │
│  - 内置 value_loss 实现                                             │
│  - init_model() 初始化                                             │
│                                                                      │
│  新版 TrainingWorker:                                               │
│  ───────────────────                                                │
│  - 通用 Worker，不内置特定 loss                                     │
│  - reset() + set_loss_fn() 设置 loss                               │
│                                                                      │
│  流程：                                                              │
│  ─────                                                              │
│  1. reset() - 初始化 TrainingWorker                                │
│  2. set_loss_fn(value_loss_) - 设置 Critic 的 loss 函数            │
│                                                                      │
│  value_loss 函数：                                                   │
│  ─────────────                                                      │
│  def value_loss(values, returns, config):                          │
│      # 计算 Critic loss                                            │
│      loss = MSE(values, returns)                                   │
│      return loss                                                   │
│                                                                      │
│  partial(value_loss, config=config):                               │
│  - 预绑定 config 参数                                               │
│  - Worker 调用时只传 values, returns                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四、创建的 WorkerGroup 属性

**init_workers 后 Trainer 拥有的属性：**

```python
self.actor_rollout_wg   # Actor + Rollout WorkerGroup
self.critic_wg          # Critic WorkerGroup (可选)
self.ref_policy_wg      # RefPolicy WorkerGroup (可选)
self.rm_wg              # RewardModel WorkerGroup (可选)
self.async_rollout_manager  # AgentLoopManager
```

---

## 五、调用时机

在 `main_ppo.py` 的 `TaskRunner.run()` 中调用：

```python
trainer = RayPPOTrainer(...)
trainer.init_workers()  # ← 这里调用
trainer.fit()
```

---

## 六、总结

| 步骤 | 操作 | 关键类/函数 |
|------|------|-------------|
| 1 | 创建资源池 | `resource_pool_manager.create_resource_pool()` |
| 2-6 | 准备 Worker 类 | `RayClassWithInitArgs` |
| 7 | 创建 WorkerGroup | `create_colocated_worker_cls`, `spawn()` |
| 8 | 初始化模型 | `init_model()` / `reset()` + `set_loss_fn()` |
| 9 | 创建 Manager | `AgentLoopManager` |

**一句话总结：**
`init_workers()` 完成 GPU 资源池创建、Worker 类包装、WorkerGroup 创建和模型初始化，建立完整的分布式训练环境。