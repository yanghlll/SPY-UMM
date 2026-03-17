# Show-o2 + Flow-GRPO 训练指南

## 目录

1. [环境准备](#1-环境准备)
2. [前置检查清单](#2-前置检查清单)
3. [单元测试：逐模块验证](#3-单元测试逐模块验证)
4. [单 GPU 调试训练](#4-单-gpu-调试训练)
5. [多 GPU 正式训练](#5-多-gpu-正式训练)
6. [三种训练模式对比](#6-三种训练模式对比)
7. [训练监控与调参](#7-训练监控与调参)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 环境准备

### 1.1 安装依赖

```bash
# 基于 Show-o2 的依赖环境
cd /nfs-stor/haolin.yang/Code/UMM/Show-o
pip install -r requirements.txt

# 核心依赖版本 (requirements.txt 中的关键包):
#   torch==2.2.1
#   transformers==4.41.1
#   accelerate==0.21.0
#   deepspeed==0.14.2
#   diffusers==0.30.1
#   omegaconf==2.3.0
#   wandb==0.17.0
#   einops==0.6.0
#   torchdiffeq  (ODE求解器，Show-o2 transport 模块依赖)

# 额外检查 torchdiffeq 是否已安装
pip install torchdiffeq
```

### 1.2 设置 PYTHONPATH

Show-o2 的模型代码不是标准 pip 包，需要将其加入 PYTHONPATH：

```bash
export PYTHONPATH="/nfs-stor/haolin.yang/Code/UMM/Show-o/show-o2:${PYTHONPATH}"
```

> 所有 run_scripts 中已自动包含此设置。手动运行时需要自行 export。

### 1.3 所需模型权重

| 模型 | 路径/ID | 来源 | 大小估计 |
|------|---------|------|----------|
| Wan2.1 VAE | `/nfs-stor/haolin.yang/models/Wan2.1_VAE.pth` | 本地文件 | ~200MB |
| Show-o2 1.5B | `showlab/show-o2-1.5B` | HuggingFace (自动下载) | ~3GB |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace (自动下载) | ~3GB |

### 1.4 数据准备

Vision-Zero CLEVR 数据集（可选，无此数据时训练会降级为 self-play）：

```
/nfs-stor/haolin.yang/Data/Vision-Zero/clevr-dataset/output/replacement_images/
├── original/       # 原始 CLEVR 图片 (civilian 角色使用)
└── modified/       # 修改后图片 (spy 角色使用)
```

---

## 2. 前置检查清单

在启动任何训练之前，按顺序执行以下检查：

```bash
cd /nfs-stor/haolin.yang/Code/UMM/SPY-UMM
export PYTHONPATH="/nfs-stor/haolin.yang/Code/UMM/Show-o/show-o2:${PYTHONPATH}"
```

### 检查 1: Python 导入

```python
python -c "
# 检查 Show-o2 模型能否导入
from models import Showo2Qwen2_5, omni_attn_mask_naive
print('[PASS] Show-o2 models imported')

# 检查 transport 模块
from transport import Sampler, create_transport
print('[PASS] Transport module imported')

# 检查 SPY-UMM 训练模块
from training import FlowGRPO, FlowGRPOConfig, PhaseController, VotingGRPO
from training import RewardWeightedFlowMatchingLoss
print('[PASS] Training modules imported')

# 检查数据模块
from data import SpyGameDataGenerator, VisionZeroDataAdapter
print('[PASS] Data modules imported')

# 检查 spy wrapper
from models.showo2_spy_wrapper import Showo2SpyWrapper
print('[PASS] Spy wrapper imported')

print('\\nAll imports OK!')
"
```

### 检查 2: 模型权重可用性

```python
python -c "
import os

vae_path = '/nfs-stor/haolin.yang/models/Wan2.1_VAE.pth'
vz_path = '/nfs-stor/haolin.yang/Data/Vision-Zero/clevr-dataset/output/replacement_images'

print(f'VAE weights: {\"FOUND\" if os.path.exists(vae_path) else \"MISSING\"} - {vae_path}')
print(f'Vision-Zero data: {\"FOUND\" if os.path.isdir(vz_path) else \"MISSING (optional)\"} - {vz_path}')

# 检查 HuggingFace 模型是否已缓存
from pathlib import Path
hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
showo_cached = any('show-o2' in str(p) for p in hf_cache.glob('*')) if hf_cache.exists() else False
qwen_cached = any('Qwen2.5-1.5B' in str(p) for p in hf_cache.glob('*')) if hf_cache.exists() else False
print(f'Show-o2 1.5B: {\"CACHED\" if showo_cached else \"WILL DOWNLOAD (~3GB)\"}')
print(f'Qwen2.5-1.5B: {\"CACHED\" if qwen_cached else \"WILL DOWNLOAD (~3GB)\"}')
"
```

### 检查 3: GPU 状况

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
```

> 最低要求: 1.5B 模型需 1x 80GB GPU (调试)；正式训练推荐 4-8x 80GB GPUs。

---

## 3. 单元测试：逐模块验证

在上训练前，先在单 GPU 上逐模块测试。以下每个测试脚本独立运行。

### 测试 1: FlowGRPO 核心算法（纯数学，不需要 GPU 模型）

```python
"""测试 Flow-GRPO 的 ODE->SDE 转换、log-prob 计算、GRPO loss"""
import torch
from training.flow_grpo import FlowGRPO, FlowGRPOConfig

# 配置
config = FlowGRPOConfig(
    sde_noise_scale=0.7,
    group_size=4,
    epsilon=0.2,
    beta=0.01,
    num_train_steps=5,   # 少量步数快速测试
    fast_mode=False,
    do_shift=False,       # 先不用 time shift 简化测试
)
flow_grpo = FlowGRPO(config)

# ---- 测试 1a: 噪声调度 ----
t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
sigma = flow_grpo.sigma_t(t)
print(f"sigma_t at t={t.tolist()}: {sigma.tolist()}")
assert sigma[0] < sigma[-1], "sigma should increase with t"
print("[PASS] Noise schedule")

# ---- 测试 1b: SDE drift ----
B, C, H, W = 2, 16, 4, 4  # 小尺寸测试
x_t = torch.randn(B, C, H, W)
t_batch = torch.tensor([0.3, 0.5])
v_theta = torch.randn(B, C, H, W)
drift = flow_grpo.sde_drift(x_t, t_batch, v_theta)
assert drift.shape == (B, C, H, W), f"Drift shape mismatch: {drift.shape}"
print("[PASS] SDE drift shape")

# ---- 测试 1c: SDE step ----
x_next, noise = flow_grpo.sde_step(x_t, t_batch, dt=0.1, v_theta=v_theta)
assert x_next.shape == (B, C, H, W)
print("[PASS] SDE step")

# ---- 测试 1d: Log probability ----
log_prob = flow_grpo.compute_step_logprob(x_t, x_next, t_batch, dt=0.1, v_theta=v_theta)
assert log_prob.shape == (B,), f"Log prob shape: {log_prob.shape}"
assert torch.isfinite(log_prob).all(), "Log prob has NaN/Inf"
print(f"[PASS] Log prob values: {log_prob.tolist()}")

# ---- 测试 1e: KL divergence ----
v_ref = torch.randn(B, C, H, W)
kl = flow_grpo.compute_step_kl(t_batch, dt=0.1, v_theta=v_theta, v_ref=v_ref)
assert kl.shape == (B,)
assert (kl >= 0).all(), "KL should be non-negative"
print(f"[PASS] KL divergence: {kl.tolist()}")

# ---- 测试 1f: 完整轨迹生成 ----
def dummy_model_fn(x, t, **kwargs):
    """模拟速度预测：返回指向原点的速度"""
    return -x * 0.1

z = torch.randn(B, C, H, W)
trajectory = flow_grpo.generate_sde_trajectory(dummy_model_fn, z, num_steps=5)
assert len(trajectory['states']) == 6, f"States count: {len(trajectory['states'])}"
assert len(trajectory['noises']) == 5
assert len(trajectory['velocities']) == 5
assert trajectory['final'].shape == (B, C, H, W)
print("[PASS] SDE trajectory generation")

# ---- 测试 1g: 轨迹 log prob ----
total_lp = flow_grpo.compute_trajectory_logprob(dummy_model_fn, trajectory)
assert total_lp.shape == (B,)
assert torch.isfinite(total_lp).all()
print(f"[PASS] Trajectory log prob: {total_lp.tolist()}")

# ---- 测试 1h: GRPO loss ----
rewards = torch.tensor([1.0, -1.0, 0.5, -0.5])
advantages = FlowGRPO.compute_advantages(rewards)
print(f"Advantages: {advantages.tolist()}")

current_lp = torch.tensor([-10.0, -12.0, -11.0, -13.0], requires_grad=True)
old_lp = torch.tensor([-10.5, -11.5, -11.5, -12.5])
result = flow_grpo.compute_grpo_loss(current_lp, old_lp, advantages)
print(f"GRPO loss: {result['loss'].item():.4f}")
print(f"Metrics: {result['metrics']}")
result['loss'].backward()
assert current_lp.grad is not None, "Gradient should flow"
print("[PASS] GRPO loss backward")

# ---- 测试 1i: Fast mode step selection ----
config_fast = FlowGRPOConfig(fast_mode=True, sde_window_size=2, sde_window_range=(0.1, 0.9))
flow_grpo_fast = FlowGRPO(config_fast)
steps = flow_grpo_fast.select_fast_steps(10, device=torch.device('cpu'))
print(f"[PASS] Fast mode selected steps: {steps}")
assert len(steps) <= 2

print("\n========== ALL FlowGRPO TESTS PASSED ==========")
```

### 测试 2: SpyGameDataGenerator（不需要 GPU）

```python
"""测试游戏数据生成"""
from data import SpyGameDataGenerator

gen = SpyGameDataGenerator(
    num_players=4,
    num_objects_min=3,
    num_objects_max=6,
    num_to_modify=2,
)

# 生成一局游戏
game = gen.generate_game(epoch=0, sample_idx=0)
print(f"Game ID: {game['game_id']}")
print(f"Num players: {game['num_players']}")
print(f"Spy player: {game['spy_player']}")
print(f"Player descriptions keys: {list(game.keys())}")

# 生成 prompt
for pid in range(1, 5):
    prompt = gen.format_generation_prompt_simple(game, pid)
    role = "SPY" if pid == game['spy_player'] else "CIV"
    print(f"  Player {pid} ({role}) prompt: {prompt[:80]}...")

# 投票 prompt
vote_prompt = gen.format_voting_prompt(game)
print(f"Voting prompt: {vote_prompt[:100]}...")

print("[PASS] Game data generation")
```

### 测试 3: Show-o2 模型加载（需要 GPU）

```python
"""测试 Show-o2 模型和 VAE 加载"""
import torch
from models import Showo2Qwen2_5, WanVAE
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import path_to_llm_name

device = torch.device("cuda:0")
dtype = torch.bfloat16

# 加载 VAE
print("Loading VAE...")
vae = WanVAE(
    vae_pth="/nfs-stor/haolin.yang/models/Wan2.1_VAE.pth",
    dtype=dtype, device=device
)
print("[PASS] VAE loaded")

# 加载 Tokenizer
print("Loading tokenizer...")
tokenizer, token_ids = get_text_tokenizer(
    "Qwen/Qwen2.5-1.5B-Instruct",
    add_showo_tokens=True,
    return_showo_token_ids=True,
    llm_name=path_to_llm_name["Qwen/Qwen2.5-1.5B-Instruct"],
)
print(f"[PASS] Tokenizer loaded, vocab size: {len(tokenizer)}")
print(f"  Token IDs: {token_ids}")

# 加载 Show-o2
print("Loading Show-o2 model...")
model = Showo2Qwen2_5.from_pretrained(
    "showlab/show-o2-1.5B", use_safetensors=False
).to(device)
print(f"[PASS] Show-o2 loaded, params: {sum(p.numel() for p in model.parameters()):,}")

# 测试 prepare_gen_input
prompts = ["A red cube on a gray floor"]
text_tokens, text_tokens_null, mod_pos, mod_pos_null = prepare_gen_input(
    prompts, tokenizer, 730,  # num_t2i_tokens (729+1 for time embed)
    token_ids['bos_id'], token_ids['eos_id'],
    token_ids['boi_id'], token_ids['eoi_id'],
    tokenizer.pad_token_id, token_ids['img_pad_id'],
    290, device  # max_text_len = 1024 - 730 - 4
)
print(f"[PASS] prepare_gen_input: text_tokens shape = {text_tokens.shape}")

# 清理
del model, vae
torch.cuda.empty_cache()
print("\n========== MODEL LOADING TESTS PASSED ==========")
```

### 测试 4: SpyWrapper 端到端生成（需要 GPU）

```python
"""测试 Spy Wrapper 的图像生成 + Flow-GRPO SDE 采样"""
import torch
from omegaconf import OmegaConf
from models import Showo2Qwen2_5, WanVAE
from models.misc import get_text_tokenizer
from models.showo2_spy_wrapper import Showo2SpyWrapper
from transport import Sampler, create_transport
from training.flow_grpo import FlowGRPO, FlowGRPOConfig
from utils import path_to_llm_name

device = torch.device("cuda:0")

# 加载 config
config = OmegaConf.load("configs/spy_umm_1.5b_flow_grpo.yaml")

# 加载模型组件
vae = WanVAE(
    vae_pth=config.model.vae_model.pretrained_model_path,
    dtype=torch.bfloat16, device=device
)

tokenizer, token_ids = get_text_tokenizer(
    config.model.showo.llm_model_path,
    add_showo_tokens=True,
    return_showo_token_ids=True,
    llm_name=path_to_llm_name[config.model.showo.llm_model_path],
)
config.model.showo.llm_vocab_size = len(tokenizer)

model = Showo2Qwen2_5.from_pretrained(
    config.model.showo.pretrained_model_path,
    use_safetensors=False
).to(device)

# Time embed 调整
if config.model.showo.add_time_embeds:
    config.dataset.preprocessing.num_mmu_image_tokens += 1
    config.dataset.preprocessing.num_t2i_image_tokens += 1

# Transport
transport = create_transport(
    path_type=config.transport.path_type,
    prediction=config.transport.prediction,
    loss_weight=config.transport.loss_weight,
    train_eps=config.transport.train_eps,
    sample_eps=config.transport.sample_eps,
    snr_type=config.transport.snr_type,
    do_shift=config.transport.do_shift,
    seq_len=config.dataset.preprocessing.num_t2i_image_tokens,
)
sampler = Sampler(transport)

# Spy Wrapper
wrapper = Showo2SpyWrapper(
    model=model, vae_model=vae,
    text_tokenizer=tokenizer, showo_token_ids=token_ids,
    transport=transport, sampler=sampler, config=config,
)
print("[PASS] SpyWrapper created")

# ---- 测试 4a: ODE 图像生成 ----
print("\nTesting ODE image generation (5 steps for speed)...")
with torch.no_grad():
    result = wrapper.generate_images(
        ["A red cube and a blue sphere"],
        guidance_scale=5.0,
        num_steps=5,  # 仅 5 步快速测试
    )
print(f"[PASS] ODE generation: {len(result['images'])} images, "
      f"latents shape: {result['latents'].shape}")

# ---- 测试 4b: SDE 图像生成 (Flow-GRPO) ----
print("\nTesting SDE image generation...")
flow_grpo_cfg = FlowGRPOConfig(
    sde_noise_scale=0.7,
    num_train_steps=3,  # 仅 3 步快速测试
    do_shift=config.transport.do_shift,
    time_shifting_factor=config.transport.time_shifting_factor,
)
flow_grpo = FlowGRPO(flow_grpo_cfg).to(device)

with torch.no_grad():
    sde_result = wrapper.generate_images_sde(
        ["A red cube and a blue sphere"],
        flow_grpo,
        num_steps=3,
        guidance_scale=0.0,
    )
print(f"[PASS] SDE generation: {len(sde_result['images'])} images, "
      f"latents shape: {sde_result['latents'].shape}")
print(f"  Trajectory states: {len(sde_result['trajectory']['states'])}")

# ---- 测试 4c: Flow-GRPO log prob 计算 ----
print("\nTesting Flow-GRPO log prob computation...")
lp = wrapper.compute_flow_grpo_logprobs(
    flow_grpo,
    sde_result['trajectory'],
    sde_result['velocity_fn_kwargs'],
)
print(f"[PASS] Log prob shape: {lp.shape}, value: {lp.item():.2f}")

# ---- 测试 4d: 带梯度的 log prob ----
print("\nTesting log prob with gradient...")
model.train()
lp_grad = wrapper.compute_flow_grpo_logprobs(
    flow_grpo,
    sde_result['trajectory'],
    sde_result['velocity_fn_kwargs'],
)
loss = -lp_grad.mean()
loss.backward()
grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f"[PASS] Gradient flows to {grad_count} parameter groups")

# 清理
model.zero_grad()
del wrapper, model, vae
torch.cuda.empty_cache()
print("\n========== SPY WRAPPER TESTS PASSED ==========")
```

### 测试 5: Reward 和 Voting 模块

```python
"""测试 reward 函数和 voting GRPO"""
from training.rewards import (
    vote_accuracy_reward, vote_format_reward,
    game_outcome_reward, compute_grpo_advantages
)
from training.grpo_voting import VotingGRPO

# ---- Reward 函数 ----
# 正确格式 + 正确投票
resp_good = "<think>Player 2's clues seem different.</think><answer>2</answer>"
assert vote_accuracy_reward(resp_good, correct_spy=2) == 1.0
assert vote_format_reward(resp_good) >= 0.8
print("[PASS] Correct vote + good format")

# 错误投票
resp_wrong = "<think>I think it's player 3.</think><answer>3</answer>"
assert vote_accuracy_reward(resp_wrong, correct_spy=2) == -1.0
print("[PASS] Wrong vote")

# 无效格式
resp_bad = "I don't know who the spy is"
assert vote_accuracy_reward(resp_bad, correct_spy=2) == -1.0
assert vote_format_reward(resp_bad) < 0.5
print("[PASS] Invalid format")

# Game outcome
outcome = game_outcome_reward(spy_caught=True, spy_player=2, num_players=4)
assert outcome['spy_reward'] == -1.0
assert outcome['civilian_reward'] == 1.0
print("[PASS] Game outcome rewards")

# GRPO advantages
advs = compute_grpo_advantages([1.0, -1.0, 0.5, -0.5])
assert abs(sum(advs)) < 0.01, "Advantages should sum to ~0"
print(f"[PASS] GRPO advantages: {advs}")

# ---- Voting GRPO loss ----
import torch
grpo = VotingGRPO(beta=0.0, epsilon=0.2)
cur_lp = torch.randn(4, 10, requires_grad=True)
old_lp = torch.randn(4, 10)
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
mask = torch.ones(4, 10)
result = grpo.compute_loss(cur_lp, old_lp, advantages, mask)
result['loss'].backward()
assert cur_lp.grad is not None
print(f"[PASS] Voting GRPO loss: {result['loss'].item():.4f}")

print("\n========== REWARD & VOTING TESTS PASSED ==========")
```

---

## 4. 单 GPU 调试训练

通过第 3 节的所有测试后，进行端到端调试。

### 4.1 Flow-GRPO 调试（推荐先测这个）

```bash
cd /nfs-stor/haolin.yang/Code/UMM/SPY-UMM
export PYTHONPATH="/nfs-stor/haolin.yang/Code/UMM/Show-o/show-o2:${PYTHONPATH}"

python train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo.yaml \
    training.max_train_steps=20 \
    training.gradient_accumulation_steps=1 \
    training.flow_grpo_group_size=2 \
    training.flow_grpo_train_steps=3 \
    experiment.save_every=10 \
    experiment.generate_every=10 \
    experiment.log_every=1 \
    experiment.name="flow-grpo-debug" \
    experiment.output_dir="output/flow-grpo-debug" \
    transport.num_inference_steps=5
```

**关键参数调整说明**：
- `flow_grpo_group_size=2`: 每个 prompt 只生成 2 张图（减少显存）
- `flow_grpo_train_steps=3`: SDE 只走 3 步（减少计算）
- `transport.num_inference_steps=5`: ODE 推理只走 5 步
- `max_train_steps=20`: 只跑 20 步看是否能跑通

### 4.2 Flow-GRPO-Fast 调试

```bash
python train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo_fast.yaml \
    training.max_train_steps=20 \
    training.gradient_accumulation_steps=1 \
    training.flow_grpo_group_size=2 \
    training.flow_grpo_train_steps=5 \
    training.sde_window_size=1 \
    experiment.save_every=10 \
    experiment.generate_every=10 \
    experiment.log_every=1 \
    experiment.name="flow-grpo-fast-debug" \
    experiment.output_dir="output/flow-grpo-fast-debug" \
    transport.num_inference_steps=5
```

### 4.3 原始 Reward-Weighted Flow 调试（对照组）

```bash
python train_spy_umm.py \
    config=configs/spy_umm_1.5b.yaml \
    training.max_train_steps=20 \
    training.gradient_accumulation_steps=1 \
    experiment.save_every=10 \
    experiment.generate_every=10 \
    experiment.log_every=1 \
    experiment.name="rw-flow-debug" \
    experiment.output_dir="output/rw-flow-debug" \
    transport.num_inference_steps=5
```

### 4.4 调试时应该检查的输出

训练应输出类似以下日志：

```
***** Running SPY-UMM Training *****
  Num players per game = 4
  Phase mode = interactive
  Flow-GRPO enabled: group_size=2, train_steps=3, fast_mode=False

Epoch: 0 Step: 1 Loss_Gen: 0.0234 Loss_Vote: 0.0000 SpyRate: 50.00% ...
Epoch: 0 Step: 2 Loss_Gen: 0.0189 Loss_Vote: 0.0000 SpyRate: 50.00% ...
```

**如果看到以下情况表示正常**：
- `Loss_Gen` 在 0.001 ~ 1.0 范围内波动
- `Loss_Vote` 在 voting phase 时非零
- `SpyRate` 初期在 20%-60% 之间
- `flow_grpo_clip_fraction` 在 0.0-0.5 之间
- `flow_grpo_mean_ratio` 在 0.5-2.0 之间

**如果出现以下情况需要排查**：
- `Loss_Gen = NaN` → 见 [常见问题 8.1](#81-nan-loss)
- `flow_grpo_mean_ratio` 爆炸 (>10 或 <0.01) → 降低学习率
- OOM → 减小 `flow_grpo_group_size` 或 `flow_grpo_train_steps`

---

## 5. 多 GPU 正式训练

### 5.1 使用 accelerate launch

```bash
cd /nfs-stor/haolin.yang/Code/UMM/SPY-UMM

# Flow-GRPO (完整版)
bash run_scripts/run_spy_umm_flow_grpo.sh

# Flow-GRPO-Fast
bash run_scripts/run_spy_umm_flow_grpo.sh configs/spy_umm_1.5b_flow_grpo_fast.yaml
```

### 5.2 自定义 GPU 数量

```bash
export PYTHONPATH="/nfs-stor/haolin.yang/Code/UMM/Show-o/show-o2:${PYTHONPATH}"
cd /nfs-stor/haolin.yang/Code/UMM/SPY-UMM

# 4 GPU 训练
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    --main_process_port=9998 \
    train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo.yaml

# 带 DeepSpeed ZeRO-2 的 8 GPU 训练
accelerate launch \
    --config_file=/nfs-stor/haolin.yang/Code/UMM/Show-o/accelerate_configs/8_gpus_deepspeed_zero2.yaml \
    train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo.yaml
```

### 5.3 正式训练推荐配置

| 配置项 | 1.5B 模型 (4x A100) | 1.5B 模型 (8x A100) |
|--------|---------------------|---------------------|
| `flow_grpo_group_size` | 4 | 8 |
| `flow_grpo_train_steps` | 10 | 10 |
| `gradient_accumulation_steps` | 4 | 2 |
| `batch_size` | 1 | 1 |
| `max_train_steps` | 10000 | 10000 |
| `learning_rate_showo` | 5e-5 | 5e-5 |
| 预计时间 | ~3-4 天 | ~2 天 |

### 5.4 使用 SLURM 提交

```bash
#!/bin/bash
#SBATCH --job-name=spy-umm-flow-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=72:00:00
#SBATCH --output=logs/flow_grpo_%j.out

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="/nfs-stor/haolin.yang/Code/UMM/Show-o/show-o2:${PYTHONPATH}"

cd /nfs-stor/haolin.yang/Code/UMM/SPY-UMM

accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --main_process_port=9998 \
    train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo.yaml
```

---

## 6. 三种训练模式对比

| | Reward-Weighted Flow | Flow-GRPO | Flow-GRPO-Fast |
|---|---|---|---|
| **Config** | `spy_umm_1.5b.yaml` | `spy_umm_1.5b_flow_grpo.yaml` | `spy_umm_1.5b_flow_grpo_fast.yaml` |
| **use_flow_grpo** | `False` | `True` | `True` |
| **fast_mode** | N/A | `False` | `True` |
| **原理** | 用 reward 加权 MSE loss | ODE→SDE + PPO-clip GRPO | 只在 1-2 个 SDE 步上做 GRPO |
| **梯度来源** | flow matching MSE | policy gradient (log-prob ratio) | policy gradient (稀疏步) |
| **每步计算量** | 1x | ~G x T_train x (较大) | ~G x 2 x (中等) |
| **理论优势** | 简单，稳定 | 直接优化策略 | 接近 Flow-GRPO 效果，5-10x 快 |
| **适用阶段** | 快速原型 | 正式训练 | 大规模训练 |

**推荐训练流程**：

```
                 调试验证                    确认可行                     规模化
Reward-Weighted ──────────> Flow-GRPO-Fast ──────────> Flow-GRPO (完整)
  (最快跑通)                  (平衡速度/效果)              (最佳效果)
```

---

## 7. 训练监控与调参

### 7.1 WandB 监控指标

训练会自动记录以下指标到 WandB：

**Loss 指标**：
- `step_loss_gen`: 生成阶段 loss（Flow-GRPO 时为 GRPO loss）
- `step_loss_vote`: 投票阶段 GRPO loss
- `step_loss_total`: 总 loss

**Flow-GRPO 指标**（仅 `use_flow_grpo=True` 时）：
- `flow_grpo_clip_fraction`: PPO clip 比例，理想值 0.05-0.3
- `flow_grpo_mean_ratio`: 重要性采样比值，应在 0.8-1.2 附近
- `flow_grpo_approx_kl`: 近似 KL 散度
- `flow_grpo_mean_logprob`: 平均 log 概率

**游戏指标**：
- `spy_detection_rate`: Spy 被识破的比例
- `mean_vote_accuracy`: 投票准确率
- `baseline_spy` / `baseline_civ`: Reward baseline (EMA)

**学习率**：
- `lr_ve`, `lr_proj`, `lr_showo`: 三组参数各自的学习率

### 7.2 超参数调优指南

#### Flow-GRPO 特有参数

| 参数 | 默认值 | 调参建议 |
|------|--------|----------|
| `sde_noise_scale` | 0.7 | 0.5-1.0。偏大→探索多但不稳定；偏小→保守但可能陷入局部最优 |
| `flow_grpo_group_size` | 4 | 越大 advantage 估计越准，但显存线性增长。4-8 是平衡点 |
| `flow_grpo_beta` | 0.01 | KL 惩罚系数。偏大→策略变化慢；偏小→可能不稳定。0.001-0.1 |
| `epsilon` | 0.2 | PPO clip 范围。标准值 0.1-0.3 |
| `flow_grpo_train_steps` | 10 | SDE 步数。论文发现 10 步和 40 步效果接近 |

#### 通用训练参数

| 参数 | 默认值 | 调参建议 |
|------|--------|----------|
| `learning_rate_showo` | 5e-5 | 若 ratio 爆炸则降到 1e-5；若收敛太慢则升到 1e-4 |
| `learning_rate_ve` | 1e-5 | 视觉编码器学习率，通常较小 |
| `max_grad_norm` | 1.0 | 梯度裁剪。Flow-GRPO 可能需要更激进 (0.5) |
| `cycle_length` | 10 | Interactive 模式下每个 phase 的步数 |
| `phase_mode` | interactive | 初期可用 "generation" 只训练生成 |

### 7.3 Checkpoint 使用

```bash
# 从 checkpoint 恢复训练
python train_spy_umm.py \
    config=configs/spy_umm_1.5b_flow_grpo.yaml \
    experiment.resume_from_checkpoint=True \
    experiment.output_dir="output/spy-umm-1.5b-flow-grpo"  # 包含 checkpoint-XXX 目录
```

### 7.4 生成样图评估

训练过程中每 `generate_every` 步会生成样图，保存在：

```
output/<experiment_name>/
├── sample_images/
│   ├── step_200.png    # 2x2 网格：4个玩家的生成图
│   ├── step_400.png
│   └── ...
├── checkpoint-500/
│   └── unwrapped_model/
├── config.yaml
└── logs/
```

同时会上传到 WandB 的 `spy_game_images` 面板。

---

## 8. 常见问题排查

### 8.1 NaN Loss

**现象**: `Loss_Gen: nan` 或训练崩溃

**排查步骤**:

```python
# 在 train_spy_umm.py 的 backward 前添加检查
if torch.isnan(total_loss) or torch.isinf(total_loss):
    logger.warning(f"NaN/Inf loss detected! gen={gen_loss.item()}, vote={vote_loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    continue  # 跳过这步
```

**常见原因**:
1. `sde_noise_scale` 过大 → 降到 0.3-0.5
2. 学习率过大 → 降低所有 lr 到 1e-5
3. bf16 精度问题 → Flow-GRPO 的 log-prob 计算在低噪声步可能溢出
   - 解决: `flow_grpo_train_steps` 从 10 降到 5, 或 `sde_window_range: [0.2, 0.8]`
4. Timestep t 接近 0 或 1 → `sigma_t` 已有 clamp(1e-4, 1-1e-4) 保护

### 8.2 OOM (显存不足)

**按优先级逐项尝试**:

1. 减小 `flow_grpo_group_size`: 4 → 2
2. 减小 `flow_grpo_train_steps`: 10 → 5
3. 开启 `flow_grpo_fast_mode: True` + `sde_window_size: 1`
4. 增大 `gradient_accumulation_steps`: 4 → 8
5. 开启 `gradient_checkpointing: True` (已默认开启)
6. 使用 DeepSpeed ZeRO-2 或 ZeRO-3

**显存估算** (1.5B 模型, bf16):
- 模型参数: ~3GB
- 优化器状态 (AdamW): ~6GB
- Flow-GRPO G=4, T=10: 每条轨迹 ~0.5GB, 共 ~8GB (4 players x 4 generations / 分时)
- 前向激活: ~4GB (有 gradient checkpointing)
- **总计**: ~21GB/GPU (单 GPU), ~6GB/GPU (8 GPU ZeRO-2)

### 8.3 Clip Fraction 过高

**现象**: `flow_grpo_clip_fraction > 0.5`

**含义**: 超过一半的 importance ratio 被 clip 了, 说明新旧策略差异过大

**解决**:
1. 降低学习率
2. 增大 `epsilon` (0.2 → 0.3)
3. 增大 `flow_grpo_beta` (加强 KL 约束)

### 8.4 Ratio 爆炸

**现象**: `flow_grpo_mean_ratio > 5` 或 `< 0.1`

**含义**: log-prob 差异过大, 导致 importance sampling 权重极端

**解决**:
1. 降低学习率 (最关键)
2. 确认 `flow_grpo_train_steps` 不要太大 (每步的 log-prob 会累加)
3. Fast mode 下只训 1-2 步, 累积误差更小

### 8.5 Spy Detection Rate 不变

**现象**: `spy_detection_rate` 长期在 25% (随机) 或 0%

**含义**: 投票阶段没有学到有效信息

**排查**:
1. 确认 `phase_mode` 包含 voting (`interactive` 或 `voting` 或 `both`)
2. 检查 `vote_temperature` 不要太低 (需要探索)
3. 查看 WandB 的 `mean_vote_accuracy` 是否有提升趋势

### 8.6 模型下载失败

```bash
# 手动预下载模型
python -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('Qwen tokenizer downloaded')
"

# 如果 HuggingFace 访问受限, 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 8.7 torchdiffeq 报错

```
ModuleNotFoundError: No module named 'torchdiffeq'
```

```bash
pip install torchdiffeq
```

> 注意: Flow-GRPO 的 SDE 采样不依赖 torchdiffeq (用的是手写 Euler-Maruyama)，
> 但 Show-o2 原始的 ODE 推理 (generate_images) 使用 torchdiffeq.odeint。

---

## 附录: 完整文件清单

```
SPY-UMM/
├── train_spy_umm.py                          # 主训练脚本
├── configs/
│   ├── spy_umm_1.5b.yaml                     # 原始 reward-weighted flow
│   ├── spy_umm_1.5b_flow_grpo.yaml           # Flow-GRPO (完整)
│   └── spy_umm_1.5b_flow_grpo_fast.yaml      # Flow-GRPO-Fast
├── run_scripts/
│   ├── run_spy_umm.sh                        # 原始训练启动
│   ├── run_spy_umm_debug.sh                  # 调试启动
│   └── run_spy_umm_flow_grpo.sh              # Flow-GRPO 启动
├── training/
│   ├── __init__.py
│   ├── flow_grpo.py                          # ★ Flow-GRPO 核心算法
│   ├── grpo_voting.py                        # 投票阶段 GRPO
│   ├── reward_weighted_flow.py               # Reward 加权 flow loss
│   ├── rewards.py                            # Reward 函数
│   └── phase_controller.py                   # 训练阶段控制器
├── models/
│   ├── __init__.py
│   └── showo2_spy_wrapper.py                 # ★ Show-o2 封装 (含 SDE 方法)
├── data/
│   ├── __init__.py
│   ├── game_data_generator.py                # 游戏数据生成
│   ├── scene_description_generator.py        # 场景描述生成
│   └── vision_zero_adapter.py                # Vision-Zero 数据适配
└── inference/
    └── generate_and_evaluate.py              # 推理评估
```
