# =============================================================================
# 模块 1: 超参数管理 (Hyperparameters Config) —— v2 纯向量射线训练版
#
# 修改记录 (v1 → v2):
#   [Fix-1] state_dim: 54 → 1610  (真实维度: 1604射线 + 6基础向量)
#   [Fix-2] behavior_name: "UGVAgent" → "UGV"  (匹配Unity场景中的m_BehaviorName)
#   [Fix-3] entropy_coef: 0.01 → 0.005  (纯向量信号清晰，降低过度探索)
#   [Fix-4] eval_interval: 20 → 10  (更频繁评估，及时发现收敛信号)
#   [Fix-5] save_interval: 50 → 20  (更频繁保存，防止灾难性遗忘时丢失最优点)
#
# 网络规模: mlp_hidden_dims=[256,256], feature_dim=128 保持不变
#   原因: 1610维输入→256隐层比例合理，射线数据结构清晰，无需放大网络
# =============================================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    # ------------------------------------------------------------------
    # [环境配置]
    # ------------------------------------------------------------------

    # Unity 可执行文件路径（None = 连接 Unity Editor）
    env_path: str = None

    worker_id: int = 0

    # [Fix-2] 与 Unity 场景中 BehaviorParameters.m_BehaviorName 对齐
    # Unity 会自动追加 "?team=0"，env_wrapper.py 用模糊匹配处理
    behavior_name: str = "UGV"

    # [Fix-1] 真实状态维度 = 射线传感器(1604) + 基础向量观测(6) = 1610
    # 关闭 Camera Sensor 后的纯向量观测总维度
    # env_wrapper.py 启动时会自动检测并打印，若不一致会给出警告
    # 禁用摄像��后首次运行，请确认控制台输出的实际维度与此一致
    state_dim: int = 1610

    # 动作空间: [ContinuousActions[0]=Steer, ContinuousActions[1]=Motor]
    action_dim: int = 2

    # ------------------------------------------------------------------
    # [PPO 核心超参数]
    # ------------------------------------------------------------------

    # 学习率 (Learning Rate)
    # 3e-4 是连续控制 PPO 的经典起点，射线观测清晰，此值合适
    lr: float = 3e-4

    # 折扣因子 γ (Discount Factor)
    # MaxStep=1000，每步时间惩罚 -1/1000，gamma=0.99 保证长视野
    gamma: float = 0.99

    # GAE λ (Generalized Advantage Estimation Lambda)
    # 0.95: bias-variance 最佳平衡，连续控制经典值
    gae_lambda: float = 0.95

    # PPO Clip 系数 ε
    clip_ratio: float = 0.2

    # [Fix-3] 熵奖励系数 (Entropy Bonus Coefficient)
    # 从 0.01 降至 0.005
    # 原因：上一次训练 Entropy 从 2.8 飙升到 4.64，说明熵正则化过强，
    #       导致策略无法稳定收敛为确定性行为。
    #       纯射线观测信息量充足，不需要强迫探索。
    entropy_coef: float = 0.005

    # Critic 损失权重
    value_loss_coef: float = 0.5

    # 梯度裁剪
    max_grad_norm: float = 0.5

    # ------------------------------------------------------------------
    # [训练流程参数]
    # ------------------------------------------------------------------

    # 每次 rollout 收集的步数
    # 4096 步适合单智能体，MaxStep=1000 时约 4 个完整 episode
    rollout_steps: int = 4096

    # Mini-batch 大小
    # 4096 步 ÷ 256 = 16 个 mini-batch，每次 ppo_epochs=10 共 160 次梯度更新
    mini_batch_size: int = 256

    # PPO 数据复用轮数
    ppo_epochs: int = 10

    # 总训练步数
    # 200万步对于 1610 维射线 + 简单 MLP 完全足够收敛
    total_timesteps: int = 2_000_000

    # ------------------------------------------------------------------
    # [网络结构参数] —— 保持不变
    # ------------------------------------------------------------------

    # MLP 隐藏层维度
    # [256, 256]: 1610→256→256→128，参数量约 50 万，合理
    # 不需要放大：射线数据是结构化信号，不像图像需要大容量网络
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # 特征提取器输出维度
    feature_dim: int = 128

    # ------------------------------------------------------------------
    # [日志与保存参数]
    # ------------------------------------------------------------------

    log_dir: str = "runs/ugv_ppo"
    checkpoint_dir: str = "checkpoints/ugv_ppo"

    # [Fix-5] 每 20 次 rollout 保存一次（约每 81920 步）
    # 更频繁保存是为了捕捉 60% Success Rate 的瞬间最优点（上次在 40 万步出现）
    save_interval: int = 20

    # [Fix-4] 每 10 次 rollout 评估一次（约每 40960 步）
    # 更频繁评估：上次错过了 40万步的 60% 成功率，因为间隔太长
    eval_interval: int = 10

    # 每次评估的 episode 数
    eval_episodes: int = 5

    # 随机种子
    seed: int = 42

    # 计算设备
    device: str = "cuda"