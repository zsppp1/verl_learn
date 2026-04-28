# verl_learn

verl (HybridFlow) RLHF 框架学习资料整理

## 目录结构

```
verl_learn/
├── HybridFlow_RLHF_Framework.pdf    # 论文原版
├── verl_intro_chinese.html          # 中文介绍
├── ppo_training_animation.html      # PPO流程动画（基础版）
├── ppo_training_animation_detailed.html  # PPO流程动画（详细版）
├── verl_architecture_guide.md       # 架构走读指南
├── main_ppo/                        # main_ppo.py 函数解析
│   ├── main_ppo_walkthrough.md
│   ├── task_runner_run_walkthrough.md
│   ├── add_actor_rollout_worker_walkthrough.md
│   ├── add_critic_worker_walkthrough.md
│   ├── add_reward_model_worker_walkthrough.md
│   ├── add_ref_policy_worker_walkthrough.md
│   ├── init_resource_pool_mgr_walkthrough.md
│   └── create_rl_dataset_sampler_walkthrough.md
├── ray_trainer/                     # ray_trainer.py 类和方法解析
│   ├── ray_trainer_overview.md
│   ├── resource_pool_manager_walkthrough.md
│   ├── helper_functions_walkthrough.md
│   ├── init_workers_walkthrough.md
│   ├── fit_walkthrough.md
│   ├── compute_methods_walkthrough.md
│   └── checkpoint_validate_walkthrough.md
├── dapo/                            # DAPO 算法解析
│   ├── dapo_integration_walkthrough.md
│   └── dapo_fit_walkthrough.md
├── technical_concepts_walkthrough.md # 技术概念速查
└── README.md
```

## 内容说明

### 1. 论文资源

- **HybridFlow_RLHF_Framework.pdf** - HybridFlow 论文原版 (EuroSys 2025)
- **verl_intro_chinese.html** - verl 中文介绍博客文章

### 2. PPO训练流程动画

- **ppo_training_animation.html** - PPO训练流程动画演示（基础版）
- **ppo_training_animation_detailed.html** - PPO训练流程动画（详细版，对应源码）

动画功能：
- 可视化PPO训练9个步骤
- 显示函数调用链和源码行号
- Worker架构联动展示
- DataProto数据流转可视化

### 3. 源码架构走读指导

- **verl_architecture_guide.md** - verl源码架构与走读指南

包含：
- 目录结构总览
- 核心架构图
- 推荐走读顺序（5个阶段）
- 核心类和函数速查表
- PPO训练流程详解
- 关键设计模式
- 配置系统介绍
- 扩展指南

### 4. main_ppo.py 函数解析

详细解析 `verl/trainer/main_ppo.py` 中的各个函数，位于 `main_ppo/` 目录：

| 文件 | 内容 |
|------|------|
| `main_ppo_walkthrough.md` | 入口函数，@hydra.main 装饰器，整体流程 |
| `task_runner_run_walkthrough.md` | TaskRunner.run() 主训练入口，7阶段完整流程 |
| `add_actor_rollout_worker_walkthrough.md` | Actor/Rollout Worker 选择和 Role 映射 |
| `add_critic_worker_walkthrough.md` | Critic Worker，新版 TrainingWorker 设计 |
| `add_reward_model_worker_walkthrough.md` | Reward Model，独立资源池机制 |
| `add_ref_policy_worker_walkthrough.md` | Reference Policy，KL 约束处理 |
| `init_resource_pool_mgr_walkthrough.md` | 资源池管理，global_pool/reward_pool |
| `create_rl_dataset_sampler_walkthrough.md` | 数据集和采样器创建 |

### 5. ray_trainer.py 类和方法解析

详细解析 `verl/trainer/ppo/ray_trainer.py` 中的核心类和方法，位于 `ray_trainer/` 目录：

| 文件 | 内容 |
|------|------|
| `ray_trainer_overview.md` | 整体架构，类结构，数据流，与 main_ppo.py 关系 |
| `resource_pool_manager_walkthrough.md` | ResourcePoolManager 类，GPU 资源池管理 |
| `helper_functions_walkthrough.md` | apply_kl_penalty, compute_advantage 等辅助函数 |
| `init_workers_walkthrough.md` | Worker 初始化流程，RayClassWithInitArgs，WorkerGroup 创建 |
| `fit_walkthrough.md` | PPO 训练主循环，9步训练流程，时序图 |
| `compute_methods_walkthrough.md` | _compute_values, _update_actor 等计算方法 |
| `checkpoint_validate_walkthrough.md` | checkpoint 保存/加载，验证评估 |

### 6. DAPO 算法解析

详细解析 DAPO 算法如何对接 verl 框架，位于 `dapo/` 目录：

| 文件 | 内容 |
|------|------|
| `dapo_integration_walkthrough.md` | main_dapo.py、dapo_ray_trainer.py 解析，继承 RayPPOTrainer，动态采样实现 |
| `dapo_fit_walkthrough.md` | RayDAPOTrainer.fit() 详细流程，动态采样循环，与 PPO 对比 |

### 7. 技术概念速查

常见技术概念和配置项的简要说明：

| 文件 | 内容 |
|------|------|
| `technical_concepts_walkthrough.md` | RayResourcePool、AgentLoopManager、non_tensor_batch、uid、ground_truth、_get_gen_batch、size_divisor、pad/unpad 等概念速查 |

## 推荐阅读顺序

### 快速入门

1. `ppo_training_animation.html` - 可视化理解 PPO 流程
2. `verl_architecture_guide.md` - 整体架构概览
3. `HybridFlow_RLHF_Framework.pdf` - 理论基础

### 深入源码 (main_ppo.py)

4. `main_ppo/main_ppo_walkthrough.md` - 入口和整体流程
5. `main_ppo/task_runner_run_walkthrough.md` - 核心训练入口
6. `main_ppo/add_actor_rollout_worker_walkthrough.md` - Worker 系统核心
7. `main_ppo/init_resource_pool_mgr_walkthrough.md` - 资源管理
8. 其他 `main_ppo/add_*_worker*.md` 文件

### 数据处理

9. `main_ppo/create_rl_dataset_sampler_walkthrough.md` - 数据加载

### 深入源码 (ray_trainer.py)

10. `ray_trainer/ray_trainer_overview.md` - Trainer 整体架构
11. `ray_trainer/init_workers_walkthrough.md` - Worker 初始化
12. `ray_trainer/fit_walkthrough.md` - PPO 训练核心循环
13. `ray_trainer/helper_functions_walkthrough.md` - Advantage 计算
14. 其他 `ray_trainer/*.md` 文件

### 算法扩展 (DAPO)

15. `dapo/dapo_integration_walkthrough.md` - DAPO 如何对接 verl

## 核心概念索引

| 概念 | 说明 | 相关文档 |
|------|------|----------|
| Worker | Ray Actor，独立进程执行任务 | `add_actor_rollout_worker_walkthrough.md` |
| WorkerGroup | 管理多个 Worker 的调度器 | `init_workers_walkthrough.md` |
| ResourcePool | GPU 资源池管理 | `resource_pool_manager_walkthrough.md` |
| Role | Worker 类型枚举 | `add_actor_rollout_worker_walkthrough.md` |
| DataProto | 数据传输协议 | `fit_walkthrough.md` |
| GAE | Generalized Advantage Estimation | `helper_functions_walkthrough.md` |
| KL Penalty | KL 散度惩罚约束 | `helper_functions_walkthrough.md` |
| PPO Clip | 策略更新裁剪 | `fit_walkthrough.md` |
| DAPO 动态采样 | 过滤全对/全错样本重新采样 | `dapo_integration_walkthrough.md` |
| RayDAPOTrainer | 继承 RayPPOTrainer 实现 DAPO | `dapo_integration_walkthrough.md` |

## 算法对比

| 算法 | 特点 | verl 配置 |
|------|------|-----------|
| PPO | 标准 PPO，需要 Critic | `adv_estimator: gae` |
| GRPO | 不需要 Critic，Group 相对优势 | `adv_estimator: grpo` |
| GSPO | 序列级别裁剪 | `policy_loss.loss_mode: gspo` |
| DAPO | 不对称裁剪 + 动态采样 + 超长惩罚 | `clip_ratio_low/clip_ratio_high` + `filter_groups` |

## 相关链接

- [verl GitHub](https://github.com/verl-project/verl)
- [verl-recipe GitHub](https://github.com/verl-project/verl-recipe)
- [HybridFlow Paper (arxiv)](https://arxiv.org/abs/2409.19256)
- [DAPO Paper (arxiv)](https://arxiv.org/abs/2503.14476)
- [DeepWiki verl 文档](https://deepwiki.com/verl-project/verl)