# 辅助函数详解

> verl/trainer/ppo/ray_trainer.py 行126-264

## 一、apply_kl_penalty 函数

### 1.1 函数定义

```python
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.
    """
```

**核心职责：**
- 计算 KL 散度：π_new vs π_ref
- 将 KL 惩罚应用到 token-level rewards
- 更新 KL controller 的 beta 值

### 1.2 代码详解

```python
response_mask = data.batch["response_mask"]
token_level_scores = data.batch["token_level_scores"]
batch_size = data.batch.batch_size[0]

# compute kl between ref_policy and current policy
kld = core_algos.kl_penalty(
    data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
)
kld = kld * response_mask
beta = kl_ctrl.value

token_level_rewards = token_level_scores - beta * kld

current_kl = masked_mean(kld, mask=response_mask, axis=-1)
current_kl = torch.mean(current_kl, dim=0).item()

kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
data.batch["token_level_rewards"] = token_level_rewards

metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

return data, metrics
```

### 1.3 KL 惩罚公式

```
┌─────────────────────────────────────────────────────────────────────┐
│                  KL 惩罚计算公式                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  token_level_rewards = token_level_scores - β × KL                  │
│                                                                      │
│  其中：                                                              │
│  ─────                                                              │
│  token_level_scores = 原始奖励分数                                  │
│  β = KL 惩罚系数 (由 kl_ctrl 控制)                                  │
│  KL = KL(π_new || π_ref) = Σ π_new × log(π_new / π_ref)            │
│                                                                      │
│  效果：                                                              │
│  ─────                                                              │
│  - 如果策略偏离 π_ref 太多 → 奖励被惩罚                             │
│  - KL controller 会动态调整 β                                       │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │   Adaptive KL Controller                                     │  │
│  │                                                               │  │
│  │   目标 KL (target_kl)                                        │  │
│  │   ─────────────                                              │  │
│  │   - 如果 current_kl > target_kl → 增大 β                    │  │
│  │   - 如果 current_kl < target_kl → 减小 β                    │  │
│  │                                                               │  │
│  │   作用：                                                      │  │
│  │   - 自动调整 KL 惩罚强度                                     │  │
│  │   - 保持策略更新在合理范围                                   │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、compute_response_mask 函数

### 2.1 函数定义

```python
def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.
    """
```

**核心职责：**
- 从完整序列的 attention_mask 中提取 response 部分

### 2.2 代码详解

```python
responses = data.batch["responses"]
response_length = responses.size(1)
attention_mask = data.batch["attention_mask"]
return attention_mask[:, -response_length:]
```

### 2.3 计算逻辑

```
┌─────────────────────────────────────────────────────────────────────┐
│                  compute_response_mask 计算逻辑                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入序列结构：                                                      │
│  ─────────                                                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  [Prompt tokens] [Response tokens] [Padding tokens]            │ │
│  │   ↑                ↑              ↑                            │ │
│  │  prompt_length    response_length  padding                     │ │
│  │                                                                │ │
│  │  attention_mask:                                               │ │
│  │  [1, 1, 1, ..., 1] [1, 1, 1, ..., 1] [0, 0, ..., 0]           │ │
│  │   ↑                ↑              ↑                            │ │
│  │  prompt部分       response部分     padding部分                  │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  提取 response_mask：                                               │
│  ─────────────────                                                  │
│                                                                      │
│  response_length = responses.size(1)                                │
│  response_mask = attention_mask[:, -response_length:]              │
│                                                                      │
│  结果：                                                              │
│  [1, 1, 1, ..., 1, 0, 0, ..., 0]                                   │
│  ↑                  ↑                                               │
│  response部分        padding部分                                    │
│                                                                      │
│  用途：                                                              │
│  ─────                                                              │
│  - 只对 response token 计算 loss                                    │
│  - 不计算 prompt token 的 KL/entropy                               │
│  - padding token mask=0，不参与计算                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、compute_advantage 函数

### 3.1 函数定义

```python
def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.
    """
```

**核心职责：**
- 根据配置选择 Advantage 估计器
- 计算 advantages 和 returns
- 将结果添加到 batch 中

### 3.2 AdvantageEstimator 类型

```
┌─────────────────────────────────────────────────────────────────────┐
│                  AdvantageEstimator 类型                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. GAE (Generalized Advantage Estimation):                        │
│  ─────────────────────────────────────────────                      │
│                                                                      │
│  A(s,a) = Σ (γλ)^t × δ_t                                            │
│                                                                      │
│  其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)                             │
│                                                                      │
│  特点：                                                              │
│  - 需要 Critic (Value Function)                                    │
│  - 需要 values 数据                                                 │
│  - 平衡偏差和方差                                                   │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  2. GRPO (Group Relative Policy Optimization):                      │
│  ──────────────────────────────────────────────                     │
│                                                                      │
│  特点：                                                              │
│  - 不需要 Critic                                                   │
│  - 基于 group 内样本相对优势                                        │
│  - 用于 DeepSeek-V2 等模型                                         │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  3. REINFORCE++:                                                     │
│  ─────────────                                                       │
│                                                                      │
│  特点：                                                              │
│  - 传统 REINFORCE 的改进版                                          │
│  - 不需要 Critic 或需要 baseline                                   │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  4. 其他估计器:                                                      │
│  ──────────                                                          │
│  - REMAX                                                            │
│  - 通过 adv_estimator_fn 动态获取                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 GAE 计算

```python
if adv_estimator == AdvantageEstimator.GAE:
    # Compute advantages and returns using Generalized Advantage Estimation (GAE)
    advantages, returns = core_algos.compute_gae_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=gamma,
        lam=lam,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

**GAE 公式详解：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  GAE 计算公式详解                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入：                                                              │
│  ─────                                                              │
│  token_level_rewards: 每个 token 的奖励                             │
│  values: Critic 预测的价值 V(s)                                     │
│  response_mask: response 部分的 mask                               │
│  gamma: 折扣因子 (默认 1.0)                                         │
│  lam: GAE lambda (默认 1.0)                                        │
│                                                                      │
│  计算步骤：                                                          │
│  ─────────                                                          │
│                                                                      │
│  Step 1: 计算 TD error                                              │
│  ───────────────────                                                │
│  δ_t = r_t + γ × V(s_{t+1}) - V(s_t)                               │
│                                                                      │
│  Step 2: 计算 GAE advantage                                        │
│  ───────────────────────                                            │
│  A_t = Σ_{l=0}^{∞} (γλ)^l × δ_{t+l}                                │
│                                                                      │
│  = δ_t + γλδ_{t+1} + (γλ)^2δ_{t+2} + ...                           │
│                                                                      │
│  Step 3: 计算 returns                                               │
│  ───────────────────                                                │
│  R_t = A_t + V(s_t)                                                 │
│                                                                      │
│  参数影响：                                                          │
│  ─────────                                                          │
│  γ=1, λ=1: 最接近真实 returns (低偏差, 高方差)                      │
│  γ=1, λ=0: TD(0) (高偏差, 低方差)                                  │
│  γ<1: 考虑未来奖励折扣                                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 GRPO 计算

```python
elif adv_estimator == AdvantageEstimator.GRPO:
    grpo_calculation_mask = data.batch["response_mask"]
    
    advantages, returns = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=grpo_calculation_mask,
        index=data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

**GRPO 特点：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  GRPO vs GAE 对比                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GRPO (Group Relative):                                             │
│  ─────────────────────                                              │
│                                                                      │
│  不需要 Critic                                                       │
│  基于同一 prompt 的多个 response group                              │
│  计算相对优势                                                        │
│                                                                      │
│  计算方式：                                                          │
│  ─────────                                                          │
│  A_i = (R_i - mean(R_group)) / std(R_group)                        │
│                                                                      │
│  其中 R_i = response_i 的总奖励                                     │
│  R_group = 同一 prompt 的所有 response 奖励                         │
│                                                                      │
│  ───────────────────────────────────────────────────────────────    │
│                                                                      │
│  GAE (需要 Critic):                                                 │
│  ───────────────────                                                │
│                                                                      │
│  需要 Critic 预测 V(s)                                              │
│  基于时间步计算 TD error                                            │
│  更精细的控制                                                        │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  GRPO 适用场景：                                               │  │
│  │  ─────────────                                                │  │
│  │  - 不想训练 Critic                                            │  │
│  │  - 有多个 response 可比较                                     │  │
│  │  - DeepSeek-V2 等大模型                                       │  │
│  │                                                               │  │
│  │  GAE 适用场景：                                                │  │
│  │  ─────────────                                                │  │
│  │  - 需要精细控制                                               │  │
│  │  - 有 Critic                                                  │  │
│  │  - 传统 PPO                                                   │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.5 其他估计器

```python
else:
    # handle all other adv estimator type other than GAE and GRPO
    adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
    
    adv_kwargs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "config": config,
    }
    if "uid" in data.non_tensor_batch:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    if "reward_baselines" in data.batch:
        adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = adv_estimator_fn(**adv_kwargs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

---

## 四、函数调用时机

```
┌─────────────────────────────────────────────────────────────────────┐
│                  辅助函数调用时机                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  fit() 训练循环中：                                                  │
│  ─────────                                                          │
│                                                                      │
│  Step 1: generate_sequences                                        │
│  Step 2: compute_reward                                            │
│  Step 3: compute old_log_prob                                      │
│  Step 4: compute values (if use_critic)                            │
│                                                                      │
│  Step 5: apply_kl_penalty (if use_kl_in_reward)                    │
│  ─────────────────────────────────────────────                      │
│  │                                                                   │
│  │  if use_kl_in_reward:                                            │
│  │      batch, kl_metrics = apply_kl_penalty(                       │
│  │          batch,                                                  │
│  │          kl_ctrl=self.kl_ctrl_in_reward,                         │
│  │          kl_penalty=self.config.algorithm.kl_penalty             │
│  │      )                                                           │
│  │                                                                   │
│  Step 6: compute_advantage                                         │
│  ─────────────────────────────                                      │
│  │                                                                   │
│  │  batch = compute_advantage(                                      │
│  │      batch,                                                      │
│  │      adv_estimator=self.config.algorithm.adv_estimator,          │
│  │      gamma=self.config.algorithm.gamma,                          │
│  │      lam=self.config.algorithm.lam,                              │
│  │  )                                                               │
│  │                                                                   │
│  Step 7: update_critic                                             │
│  Step 8: update_actor                                              │
│                                                                      │
│  compute_response_mask 在 compute_advantage 内部调用：              │
│  ─────────────────────────────────────────────────────              │
│  if "response_mask" not in data.batch.keys():                      │
│      data.batch["response_mask"] = compute_response_mask(data)     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 五、总结

| 函数 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `apply_kl_penalty` | KL惩罚计算 | data, kl_ctrl | data + KL metrics |
| `compute_response_mask` | 提取response mask | data | response_mask tensor |
| `compute_advantage` | 计算advantage | data, estimator | data + advantages |

**一句话总结：**
三个辅助函数分别处理 KL 约束、序列掩码提取和 Advantage 估计，是 PPO 算法核心计算的重要组成部分。