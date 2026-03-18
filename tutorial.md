# SPY-UMM 生成优化 Tutorial（详细代码解析）

## Context

SPY-UMM 通过 **"谁是卧底" 自博弈游戏** 对 Show-o2（统一多模态模型）的 text-to-image 生成进行 post-training 优化。核心思想：4 个玩家根据场景描述生成图像，其中 1 个 spy 收到了修改过的描述。模型需要学会：**生成尽可能忠实于描述的图像**（让 spy 的图像不被识别，或让 civilian 的图像和其他人一致）。

生成优化有 **两条路径**：
1. **Reward-Weighted Flow Matching**（简单路径）—— 当前正在运行
2. **Flow-GRPO**（高级路径）—— ODE→SDE 转换 + PPO-clip 策略梯度

---

## 一、整体训练循环（train_spy_umm.py）

```
┌──────────────────────────────────────────────────────────────┐
│  每一步训练 (1 game = 1 training step)                       │
│                                                              │
│  Step A: 生成游戏数据                                         │
│    └─ SpyGameDataGenerator.generate_game()                   │
│    └─ 输出: 4个玩家的场景描述 (spy拿到修改版)                    │
│                                                              │
│  Step B: 判断当前训练阶段                                     │
│    └─ PhaseController: generation / voting / interactive      │
│                                                              │
│  Step C: 生成图像 (torch.no_grad)                             │
│    └─ spy_wrapper.generate_images() → ODE采样                 │
│    └─ 输出: latents [4, 16, 54, 54], PIL images [4]           │
│                                                              │
│  Steps D-F: 投票阶段 (如果 train_vote)                        │
│    └─ 模型看所有图像 → 投票识别 spy                             │
│    └─ GRPO 优化投票文本生成                                    │
│                                                              │
│  Steps G-I: 生成阶段 (如果 train_gen)                         │
│    └─ G: 根据投票结果计算 game reward                          │
│    └─ H-I: 用 reward 优化图像生成                              │
│        ├─ Flow-GRPO 路径: SDE采样 → log-prob → PPO-clip loss  │
│        └─ 或 RW-Flow 路径: reward-weighted MSE loss           │
│                                                              │
│  Step J: 合并 loss → backward → optimizer.step()              │
│    └─ total_loss = gen_coeff * gen_loss + vote_coeff * vote_loss│
└──────────────────────────────────────────────────────────────┘
```

### 关键文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `train_spy_umm.py` | 915 | 主训练循环 |
| `models/showo2_spy_wrapper.py` | ~830 | 模型包装器（生成/理解/loss计算） |
| `training/flow_grpo.py` | 634 | Flow-GRPO 算法 |
| `training/reward_weighted_flow.py` | 180 | Reward-Weighted Flow Matching |
| `training/grpo_voting.py` | 153 | GRPO for 投票阶段 |
| `training/phase_controller.py` | 58 | 交互式阶段切换 |
| `training/rewards.py` | 149 | Reward 函数 |
| `data/game_data_generator.py` | 348 | 游戏数据 + reward 计算 |
| `data/scene_description_generator.py` | 215 | CLEVR 风格场景描述生成 |
| `data/vision_zero_adapter.py` | 170 | Vision-Zero 数据集适配 |

---

## 二、图像生成过程（Step C）

**文件:** `models/showo2_spy_wrapper.py` → `generate_images()`

### 2.1 流程

```python
# 1. 文本 tokenization
batch_text_tokens, batch_text_tokens_null, batch_mod_pos, _ = prepare_gen_input(
    prompts, tokenizer, num_t2i_tokens=730,  # 729 image + 1 time embed
    bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id, max_text_len, device
)
# batch_text_tokens: [B, seq_len] — 条件文本 tokens
# batch_text_tokens_null: [B, seq_len] — 无条件 tokens (CFG用)
# batch_mod_pos: [B, num_images, 2] — (offset, length) 图像位置

# 2. 初始噪声
z = torch.randn(B, 16, 54, 54)  # C=16, H=27*2, W=27*2 (patch_size=2)

# 3. Classifier-Free Guidance 拼接
z = torch.cat([z, z], dim=0)  # [2B, 16, 54, 54]
text_tokens = torch.cat([cond, uncond], dim=0)  # [2B, seq_len]

# 4. 注意力 mask
block_mask = omni_attn_mask_naive(2B, 1024, modality_positions, device)

# 5. ODE 采样: 从 t=0 (噪声) 到 t=1 (图像)
sample_fn = sampler.sample_ode(method='euler', num_steps=50, ...)
samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
# samples: [2B, 16, 54, 54] → 取前半 [B, 16, 54, 54]

# 6. VAE 解码
images_tensor = vae.batch_decode(samples.unsqueeze(2)).squeeze(2)
# [B, 3, 432, 432] RGB
```

### 2.2 ODE 采样细节 (Show-o2 transport)

```
路径: x_t = (1-t) * x_0 + t * x_1   (线性插值)
速度: v = x_1 - x_0                  (常数速度目标)

ODE:  dx/dt = v_θ(x_t, t, text)     (确定性)
求解: x_T = x_0 + ∫₀¹ v_θ(x_t, t) dt

时间偏移 (time shifting):
  t → μ / (μ + (1/t - 1)^σ)
  其中 μ 由 seq_len 决定 (time_shifting_factor=3.0), σ=1
  效果: 在噪声较大的区域（t 较小）花更多采样步数
```

---

## 三、Reward 计算（Step G）

**文件:** `data/game_data_generator.py`

### 3.1 游戏结果判定

```python
def calculate_game_rewards(game_data, player_votes):
    # 统计每个玩家获得的票数
    vote_counts = {pid: 0 for pid in range(1, num_players+1)}
    for vote in player_votes:
        if vote and vote.get('voted_spy'):
            vote_counts[vote['voted_spy']] += 1

    # 判定 spy 是否被抓 (majority vote)
    spy_caught = vote_counts[spy_player] > num_players // 2

    # 计算 per-player reward
    for pid in range(1, num_players+1):
        if pid == spy_player:
            # Spy: 隐匿分 + 误导分
            stealth = 1.0 - (被投票数 / (N-1))
            mislead = 0.5 * 误导他人数 / (N-1)
            reward = clamp(stealth + mislead, [-1, 1])
        else:
            # Civilian: 投票准确分 - 被怀疑惩罚
            accuracy = 1.0 if 投对了 else -1.0
            penalty = -0.6 * (自己被投数 / (N-1))
            reward = clamp(accuracy + penalty, [-1, 1])
```

### 3.2 转换为生成 reward

```python
def compute_generation_rewards(game_outcome):
    spy_caught = game_outcome['spy_caught']

    for pid in range(1, num_players+1):
        if pid == spy_player:
            # Spy 生成目标: 图像尽量像 civilian
            gen_reward = -1.0 if spy_caught else 1.0
        else:
            # Civilian 生成目标: 图像忠实于描述
            gen_reward = 1.0 if spy_caught else -0.5

    # 例: spy(P2)被抓 → rewards = [1.0, -1.0, 1.0, 1.0]
    # 例: spy(P2)逃脱 → rewards = [-0.5, 1.0, -0.5, -0.5]
```

**核心逻辑：**
- Spy 被抓 → 惩罚 spy 的生成（图太不一样了），奖励 civilian（图够忠实）
- Spy 逃脱 → 奖励 spy 的生成（骗过了大家），惩罚 civilian（图可能不够好）

---

## 四、路径 A: Reward-Weighted Flow Matching（当前使用）

**文件:** `training/reward_weighted_flow.py` + `models/showo2_spy_wrapper.py`

### 4.1 原理

标准 flow matching 的训练目标是：
```
L = E_t [ ||v_θ(x_t, t) - u_t||² ]
```
其中 `u_t = x_1 - x_0` 是目标速度。

Reward-weighted 版本在每个样本的 loss 前乘以 reward weight：
```
L = Σ_i  w_i * ||v_θ(x_t^i, t) - u_t^i||²  /  Σ_i w_i
```

**直觉:** 生成好的玩家（reward 高）→ 更大的 weight → 模型更多学习这些样本的模式。

### 4.2 Weight 计算 (RewardWeightedFlowMatchingLoss)

```python
class RewardWeightedFlowMatchingLoss:
    def __init__(self, reward_baseline_ema=0.9, reward_clamp_min=0.0):
        self.baseline_spy = 0.0   # Spy 的 EMA baseline
        self.baseline_civ = 0.0   # Civilian 的 EMA baseline

    def compute_weights(self, rewards, is_spy):
        weights = torch.zeros(B)

        # Spy: advantage = reward - baseline_spy
        spy_adv = rewards[is_spy] - self.baseline_spy
        weights[is_spy] = spy_adv
        self.baseline_spy = 0.9 * self.baseline_spy + 0.1 * rewards[is_spy].mean()

        # Civilian: advantage = reward - baseline_civ
        civ_adv = rewards[~is_spy] - self.baseline_civ
        weights[~is_spy] = civ_adv
        self.baseline_civ = 0.9 * self.baseline_civ + 0.1 * rewards[~is_spy].mean()

        # 归一化 + 截断为非负
        weights = weights / std(weights)
        weights = weights.clamp(min=0.0)  # 只强化好结果

        return weights  # [B]
```

**例子:**
```
rewards = [1.0, -1.0, 1.0, 1.0]  (spy被抓)
is_spy  = [F,    T,    F,   F  ]

spy_adv  = [-1.0 - 0.0] = [-1.0] → clamp → 0.0 (spy不被强化)
civ_adv  = [1.0-0, 1.0-0, 1.0-0] = [1.0, 1.0, 1.0] → 标准化 → [1.0, 1.0, 1.0]

最终 weights = [1.0, 0.0, 1.0, 1.0]
→ Spy 的图像不参与训练, civilian 的图像正常训练
```

### 4.3 Flow Loss 计算 (showo2_spy_wrapper.compute_flow_loss)

```python
def compute_flow_loss(self, prompts, target_latents, reward_weights):
    B = len(prompts)

    # 1. Tokenize
    batch_text_tokens, _, batch_mod_pos, _ = prepare_gen_input(...)

    # 2. 采样时间步 + 构造噪声版本
    for i in range(B):
        t_i, x0_i, x1_i = transport.sample(target_latents[i:i+1])
        # t_i: [1] 时间步 (lognorm 采样)
        # x0_i: [1,16,54,54] 随机噪声
        # x1_i: [1,16,54,54] = target_latents[i] (目标)

        t, xt, ut = path_sampler.plan(t_i, x0_i, x1_i)
        # xt = (1-t)*x0 + t*x1  — 噪声化的 latent
        # ut = x1 - x0          — 速度目标

    # 3. 构造 image_masks [B, max_seq_len=1024]
    image_masks = zeros(B, 1024)
    for offset, length in modality_positions:
        image_masks[i, offset:offset+length] = 1

    # 4. 模型前向 (带 labels → 返回内置 loss)
    logits, loss_flow = model(
        text_tokens=batch_text_tokens,  # [B, seq_len]
        image_latents=xt,               # [B, 16, 54, 54] 噪声化
        t=t,                            # [B] 时间步
        image_labels=ut,                # [B, 16, 54, 54] 速度目标
        image_masks=image_masks,        # [B, 1024]
        ...
    )
    # 模型内部: patchify ut → [B, 729, 64]
    # 模型内部: v_pred → MSE(v_pred, ut_patchified)

    # 5. 计算 per-sample loss (额外的无标签前向)
    _, v_pred = model(... image_labels=None ...)
    # v_pred: [B, 16, 54, 54] (unpatchified)
    per_sample_loss = MSE(v_pred, ut).mean(dim=[1,2,3])  # [B]

    # 6. 加权
    weighted_loss = (reward_weights * per_sample_loss).sum() / reward_weights.sum()

    return {'loss': weighted_loss, 'per_sample_loss': per_sample_loss}
```

### 4.4 在 train_spy_umm.py 中的调用

```python
# train_spy_umm.py (reward-weighted flow 分支)

# 获取 target latents
if vz_adapter is not None:
    target_latents = vz_adapter.get_target_latents(game_data, device)
else:
    target_latents = generated_latents.detach()  # 自博弈: 用自己生成的图

# 计算 reward weights
reward_weights = rw_flow_loss.compute_weights(reward_tensor, is_spy)

# 计算加权 flow matching loss
flow_result = spy_wrapper.compute_flow_loss(
    prompts,
    target_latents=target_latents,
    reward_weights=reward_weights,
)
gen_loss = flow_result['loss']
```

---

## 五、路径 B: Flow-GRPO（高级路径，显存需求大）

**文件:** `training/flow_grpo.py` + `models/showo2_spy_wrapper.py`

### 5.1 核心思想

Flow-GRPO 将 RLHF 的策略梯度方法（PPO/GRPO）应用于**连续空间**的 flow matching 生成。

**挑战:** 标准 GRPO 需要离散 token 的 log probability。但 Show-o2 的图像生成是通过连续 ODE 实现的，没有"采样"步骤，无法定义 log-prob。

**解决方案:** 将确定性 ODE 转换为等价的随机 SDE，这样每步转移是 Gaussian，可以计算 log-prob。

### 5.2 数学: ODE → SDE 转换

**原始 ODE:**
```
dx = v_θ(x, t) dt     (确定性, 无 log-prob)
```

**转换后的 SDE:**
```
dx = f(x, t) dt + σ_t dW    (随机, 有 log-prob)

其中:
  σ_t = a · √(t/(1-t))        噪声调度, a=0.7
  f(x,t) = v_θ + σ_t²/(2t) · (x + (1-t)·v_θ)   修正后的漂移
```

**关键性质:**
- SDE 和 ODE 有**相同的边际分布** p(x_t)
- 每步 SDE 转移是 **Gaussian**: x_{t+dt} ~ N(mean, var)
  - mean = x_t + f(x_t, t) · dt
  - var = σ_t² · dt
- 因此可以计算**精确的 log-probability**

### 5.3 噪声调度 σ_t

```python
# flow_grpo.py
def sigma_t(self, t):
    """σ_t = a * sqrt(t / (1-t))"""
    a = self.config.sde_noise_scale  # 0.7
    t_safe = t.clamp(min=1e-4, max=1-1e-4)
    return a * torch.sqrt(t_safe / (1.0 - t_safe))

def sigma_t_sq(self, t):
    """σ_t² = a² * t / (1-t)"""
    a = self.config.sde_noise_scale
    t_safe = t.clamp(min=1e-4, max=1-1e-4)
    return (a ** 2) * t_safe / (1.0 - t_safe)
```

**特性:**
```
t=0.05 → σ≈0.16  (early: 低噪声)
t=0.50 → σ=0.70  (mid: 中等噪声)
t=0.95 → σ≈3.05  (late: 高噪声)
```

### 5.4 SDE 步进 (Euler-Maruyama)

```python
# flow_grpo.py
def sde_step(self, x_t, t, dt, v_theta, noise=None):
    """
    x_{t+dt} = x_t + f(x_t,t)·dt + σ_t·√dt·ε

    其中 f(x,t) = v_θ + σ²/(2t)·(x + (1-t)·v_θ)
    """
    drift = self.sde_drift(x_t, t, v_theta)   # [B,C,H,W]
    sigma = self.sigma_t(t).view(-1,1,1,1)     # [B,1,1,1]

    if noise is None:
        noise = torch.randn_like(x_t)

    x_next = x_t + drift * dt + sigma * sqrt(|dt|) * noise
    return x_next, noise
```

### 5.5 Log-Probability 计算

```python
# flow_grpo.py
def compute_step_logprob(self, x_t, x_next, t, dt, v_theta):
    """
    log π_θ(x_{t+dt} | x_t) = log N(x_{t+dt}; mean, var)

    mean = x_t + f(x_t,t)·dt
    var  = σ_t²·dt
    """
    drift = self.sde_drift(x_t, t, v_theta)
    sigma = self.sigma_t(t).view(-1,1,1,1)

    mean = x_t + drift * dt          # [B,C,H,W]
    var = (sigma ** 2) * abs(dt)     # [B,1,1,1]

    diff = x_next - mean             # [B,C,H,W]
    D = diff[0].numel()              # 16*54*54 = 46656 维

    # Gaussian log-prob
    log_prob = -0.5 * (diff**2 / var).sum(dim=[1,2,3])  # [B]
    log_prob -= 0.5 * D * torch.log(2*π*var.squeeze())   # 归一化常数

    return log_prob  # [B]
```

### 5.6 轨迹生成

```python
# flow_grpo.py
def generate_sde_trajectory(self, model_fn, z, num_steps=10):
    """生成完整 SDE 轨迹"""

    # 时间步调度: [0, 0.1, 0.2, ..., 1.0] + time_shifting
    timesteps = linspace(0, 1, num_steps+1)  # [11]
    if do_shift:
        timesteps = timesteps / (timesteps + factor - factor * timesteps)

    states = [z]      # 初始噪声 x_0
    noises = []       # 每步使用的随机噪声
    velocities = []   # 每步的速度预测

    x_t = z
    for i in range(num_steps):  # 10 步
        v_theta = model_fn(x_t, t_i)          # 模型预测速度
        noise_i = torch.randn_like(x_t)        # 随机噪声
        x_next, _ = sde_step(x_t, t_i, dt_i, v_theta, noise=noise_i)

        states.append(x_next.detach())
        noises.append(noise_i)
        velocities.append(v_theta.detach())
        x_t = x_next.detach()

    return {
        'states': states,        # [11] 个 [1,16,54,54]
        'noises': noises,        # [10] 个 [1,16,54,54]
        'velocities': velocities, # [10] 个 [1,16,54,54]
        'timesteps': timesteps,  # [11]
        'final': x_t             # [1,16,54,54]
    }
```

### 5.7 GRPO Loss

```python
# flow_grpo.py
def compute_grpo_loss(self, current_logprobs, old_logprobs, advantages, kl_values=None):
    """
    PPO-clip loss:
    J = (1/G) Σ min(r·Â, clip(r, 1-ε, 1+ε)·Â) - β·D_KL

    r = exp(log_new - log_old)    importance ratio
    Â = (R - mean(R)) / std(R)   group-relative advantage
    """
    log_ratio = (current_logprobs - old_logprobs).clamp(-10, 10)
    ratio = exp(log_ratio)                    # [G]

    clipped = ratio.clamp(1-0.2, 1+0.2)      # PPO clip, ε=0.2

    surr1 = ratio * advantages
    surr2 = clipped * advantages
    policy_loss = -min(surr1, surr2).mean()   # 取 min 更保守

    kl_loss = 0.01 * kl_values.mean() if kl_values else 0

    total_loss = policy_loss + kl_loss
    return {'loss': total_loss, 'metrics': {...}}
```

### 5.8 完整 Flow-GRPO 训练流程 (train_spy_umm.py)

```
Phase 1: 生成轨迹 (torch.no_grad)
  对每个 player (4个) × 每个 group (G=4):
    1. SDE 采样生成 1 条轨迹 (T=10 步)
    2. 计算 old_logprob (frozen)
    → 总共 16 条轨迹, 16 个 old_logprob

Phase 2: 计算 group-relative advantages
  rewards = [R1,R1,R1,R1, R2,R2,R2,R2, R3,R3,R3,R3, R4,R4,R4,R4]
  advantages = (rewards - mean) / std   # [16]

  关键: spy 和 civilian 的 reward 不同 → 产生方差 → 有意义的 advantage

Phase 3: 梯度计算 (with grad)
  对每条轨迹:
    1. 重新计算 current_logprob (有梯度)
    2. 计算 KL(current || reference)

  汇总后计算 GRPO loss:
    loss = PPO_clip(current_lp, old_lp, advantages) + β * KL

Backward: loss.backward() → 梯度流过模型的速度预测头
```

### 5.9 Flow-GRPO-Fast 模式

```python
# 不训练所有 T=10 步, 只训练其中 2 步 (sde_window_size=2)
# 随机选取 t ∈ [0.1, 0.9] 范围内的 2 个步

def select_fast_steps(self, num_steps, device):
    t_min, t_max = 0.1, 0.9
    step_min = max(1, int(0.1 * 10))  # = 1
    step_max = min(9, int(0.9 * 10))  # = 9

    # 随机选 2 步, 例如 [3, 7]
    indices = randperm(8)[:2] + 1
    return sorted(indices)  # [3, 7]

# 效果: 只计算 step 3 和 step 7 的 log-prob
# 速度提升 5-10x, reward 损失 negligible
```

---

## 六、两条路径对比

| 维度 | Reward-Weighted Flow | Flow-GRPO |
|------|---------------------|-----------|
| **原理** | 在 MSE loss 前乘 reward weight | PPO-clip 策略梯度 |
| **前向次数/步** | 2 次 (1次带label + 1次不带) | 2×(N×G) 次 (old + new log-prob) |
| **显存** | ~30 GB (单卡可跑) | ~140 GB+ (OOM, 需多卡) |
| **数学严谨性** | 启发式 (直接缩放 loss) | 理论上正确 (策略梯度定理) |
| **训练稳定性** | 简单稳定 | PPO clip + KL 约束 |
| **收敛速度** | 较慢 | 较快 (proper credit assignment) |
| **Fast 模式** | N/A | 5-10x 加速 |

---

## 七、关键 Tensor 形状追踪

```
=== 输入 ===
prompts: [str] × 4

=== 图像生成 (ODE) ===
z:                [4, 16, 54, 54]  → CFG: [8, 16, 54, 54]
samples:          [4, 16, 54, 54]  (取前半)
generated_latents:[4, 16, 54, 54]
images_tensor:    [4, 3, 432, 432] (VAE 解码后)

=== Flow Matching Training ===
target_latents:   [4, 16, 54, 54]
t:                [4]             (时间步)
xt:               [4, 16, 54, 54] (噪声化 latent)
ut:               [4, 16, 54, 54] (速度目标)
v_pred:           [4, 16, 54, 54] (模型预测)
per_sample_loss:  [4]
reward_weights:   [4]
weighted_loss:    scalar

=== Flow-GRPO (per trajectory) ===
z_init:           [1, 16, 54, 54]
trajectory states:[11] × [1, 16, 54, 54]  (T+1=11步)
trajectory noises:[10] × [1, 16, 54, 54]  (T=10步)
logprob:          [1]    (scalar per trajectory)

=== Flow-GRPO (aggregated) ===
flat_rewards:     [16]   (4 players × 4 group)
advantages:       [16]
current_logprobs: [16]
old_logprobs:     [16]
kl_tensor:        [16]
grpo_loss:        scalar
```

---

## 八、投票阶段（Steps D-F）

**文件:** `training/grpo_voting.py` + `training/rewards.py`

### 8.1 投票流程

```python
# Step D: 生成 G 个投票 completion
for g in range(num_generations):  # G=4
    response = spy_wrapper.judge_vote(
        image_latents_list,     # 所有玩家的图像 latent
        voting_prompt,          # 投票提示词
        max_new_tokens=512,
        temperature=1.0,
    )
    # response 格式: <think>分析推理</think><answer>PLAYER_NUMBER</answer>

    # 计算 reward
    acc_reward = vote_accuracy_reward(response, correct_spy)  # ±1.0
    fmt_reward = vote_format_reward(response)                  # 0.0~1.0
    total_reward = acc_reward + fmt_reward
```

### 8.2 投票 GRPO Loss

```python
# Step E-F: GRPO 优化投票质量
advantages = (rewards - mean) / std     # [G] group-relative

# Per-token importance ratio
ratio = exp(current_logprobs - old_logprobs)  # [G, L]
clipped_ratio = clamp(ratio, 1-ε, 1+ε)

# PPO-clip per-token loss, masked by completion length
per_token_loss = -min(ratio * A, clipped_ratio * A)
vote_loss = masked_mean(per_token_loss, completion_mask)
```

---

## 九、配置文件说明

### spy_umm_1.5b.yaml（Interactive 模式基线）

```yaml
# 模型
model:
  showo:
    pretrained_model_path: "showlab/show-o2-1.5B"
    llm_model_path: "Qwen/Qwen2.5-1.5B-Instruct"
    hidden_size: 1536
    image_latent_dim: 16
    patch_size: 2
    frozen_params: ['und_trans', 'image_embedder_und', 'position_embedding']

# 游戏
game:
  num_players: 4
  num_objects_min: 3
  num_objects_max: 6
  num_objects_to_modify: 2

# 训练
training:
  phase_mode: "interactive"    # 生成↔投票交替
  cycle_length: 10             # 每10步切换阶段
  gen_loss_coeff: 1.0
  vote_loss_coeff: 1.0
  epsilon: 0.2                 # PPO clip
  max_train_steps: 10000

# 优化器（3组不同学习率）
optimizer:
  params:
    learning_rate_ve: 0.00001     # 视觉编码器
    learning_rate_proj: 0.00005   # 投影层/扩散头
    learning_rate_showo: 0.00005  # LLM backbone

# Flow matching
transport:
  path_type: "Linear"
  prediction: "velocity"
  guidance_scale: 5.0
  num_inference_steps: 50
  do_shift: True
  time_shifting_factor: 3.0
```

### Flow-GRPO 额外配置

```yaml
training:
  use_flow_grpo: True
  flow_grpo_group_size: 4      # G: 每个 prompt 生成4张图
  flow_grpo_train_steps: 10    # T: SDE 去噪步数
  flow_grpo_beta: 0.01         # KL 惩罚系数
  sde_noise_scale: 0.7         # σ_t 中的 a 参数
  # Fast 模式
  flow_grpo_fast_mode: True
  sde_window_size: 2           # 只训练2步
  sde_window_range: [0.1, 0.9] # 步选择范围
```

---

## 十、交互式阶段控制

**文件:** `training/phase_controller.py`

```python
class PhaseController:
    """
    interactive 模式: 每 cycle_length 步切换一次

    步 0-9:   训练 generation（图像生成质量）
    步 10-19: 训练 voting（投票准确度）
    步 20-29: 训练 generation
    ...交替进行
    """
    def get_active_phase(self, global_step):
        total_cycle = self.cycle_length * 2  # = 20
        position = global_step % total_cycle
        if position < self.cycle_length:
            return 'generation'
        else:
            return 'voting'
```

**直觉:** 先让模型学会投票识别 spy，再让模型利用投票反馈改进生成，循环提升。

---

## 十一、当前训练状态

当前正在用 **Reward-Weighted Flow Matching** 模式在 4×H200 上训练：
- Config: `spy_umm_1.5b.yaml`
- 4 players, interactive mode (generation ↔ voting 交替)
- 已成功通过 30+ steps, Loss_Gen ∈ [0.03, 0.18]
- WandB: https://wandb.ai/fadliaulawi/spy-umm

Flow-GRPO 模式因显存限制（每卡 140GB 不够）暂未启用。需要：
- 减少 group_size (4→2)
- 减少 num_players (4→2)
- 或使用 DeepSpeed ZeRO-3 分片

---

## 十二、验证方法

```bash
# 1. 查看训练日志
tail -f output/train_rw_flow.log | grep "Step:"

# 2. 检查 Loss_Gen 是否下降: 应从 ~0.05 逐步下降
grep "Loss_Gen" output/train_rw_flow.log | tail -20

# 3. 检查 SpyRate: spy 检测率应逐步提升
grep "SpyRate" output/train_rw_flow.log | tail -20

# 4. WandB dashboard 看 loss 曲线和生成图像样本
```

---

## 附录：完整代码目录结构

```
SPY-UMM/
├── train_spy_umm.py              # 主训练脚本 (915行)
├── tutorial.md                   # 本文档
├── TRAINING_GUIDE.md             # 训练操作指南
├── setup_env.sh                  # 环境安装脚本
│
├── configs/
│   ├── spy_umm_1.5b.yaml                  # Interactive 基线
│   ├── spy_umm_1.5b_flow_grpo.yaml        # Full Flow-GRPO
│   └── spy_umm_1.5b_flow_grpo_fast.yaml   # Flow-GRPO-Fast
│
├── models/
│   ├── __init__.py
│   └── showo2_spy_wrapper.py     # Show-o2 包装器 (~830行)
│
├── training/
│   ├── __init__.py
│   ├── flow_grpo.py              # Flow-GRPO 算法 (634行)
│   ├── reward_weighted_flow.py   # RW-Flow Matching (180行)
│   ├── grpo_voting.py            # 投票 GRPO (153行)
│   ├── phase_controller.py       # 阶段控制 (58行)
│   └── rewards.py                # Reward 函数 (149行)
│
├── data/
│   ├── __init__.py
│   ├── game_data_generator.py           # 游戏编排 (348行)
│   ├── scene_description_generator.py   # CLEVR 场景生成 (215行)
│   └── vision_zero_adapter.py           # Vision-Zero 适配 (170行)
│
├── inference/
│   └── generate_and_evaluate.py  # 推理评估脚本
│
├── run_scripts/
│   ├── run_spy_umm.sh                    # 多卡训练
│   ├── run_spy_umm_debug.sh              # 单卡调试
│   ├── run_spy_umm_flow_grpo.sh          # Flow-GRPO 训练
│   └── run_spy_umm_flow_grpo_debug.sh    # Flow-GRPO 调试
│
└── output/                       # 训练输出 & checkpoints
    └── spy-umm-1.5b-interactive/
        ├── checkpoint-500/
        └── checkpoint-1000/
```
