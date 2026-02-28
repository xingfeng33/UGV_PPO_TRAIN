# =============================================================================
# 模块 1: 超参数管理 (Hyperparameters Config) —— v4
#
# 修改记录 (v3 → v4):
#   [Fix-1] checkpoint_dir 移除了硬编码的根目录，
#           改为只存储根目录前缀 "checkpoints/ugv_ppo"。
#           实际的 run_name 子目录由 train.py 在运行时动态拼接，
#           即：checkpoints/ugv_ppo/{run_name}/
#           这样不同实验的模型文件天然隔离，不会互相覆盖。
#
#   [Fix-2] entropy_coef 恢复至 0.02（v3 已修复，保持不变）
#   [Fix-3] lr 保持 1e-4（v3 已修复，保持不变）
#   [Fix-4] ppo_epochs 保持 6（v3 已修复，保持不变）
# =============================================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:

    # ------------------------------------------------------------------
    # [环境配置]
    # ------------------------------------------------------------------
    env_path: str = None
    worker_id: int = 0
    behavior_name: str = "UGV"
    state_dim: int = 1610
    action_dim: int = 2

    # ------------------------------------------------------------------
    # [PPO 核心超参数]
    # ------------------------------------------------------------------
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.02
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ------------------------------------------------------------------
    # [训练流程参数]
    # ------------------------------------------------------------------
    rollout_steps: int = 8192
    mini_batch_size: int = 512
    ppo_epochs: int = 6
    total_timesteps: int = 5_000_000

    # ------------------------------------------------------------------
    # [网络结构参数]
    # ------------------------------------------------------------------
    # 改后：扩大到 [512, 512, 256]，feature_dim 改为 256
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    feature_dim: int = 256

    # ------------------------------------------------------------------
    # [日志与保存参数]
    # ------------------------------------------------------------------
    log_dir: str = "runs/ugv_ppo"

    # [Fix-1] 只存储根目录前缀，不包含 run_name。
    # train.py 会在运行时将其拼接为 checkpoints/ugv_ppo/{run_name}/
    # 这样每次实验的模型自动存入独立子文件夹，不会互相覆盖。
    checkpoint_dir: str = "checkpoints/ugv_ppo"

    save_interval: int = 25
    eval_interval: int = 10
    eval_episodes: int = 20
    seed: int = 42
    device: str = "cuda"