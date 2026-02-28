# =============================================================================
# 模块 3: 网络结构 (Networks)
#
# 设计哲学（重要！请在论文中说明）：
#
#   采用 "特征提取器 + 独立头部" 的高度解耦架构：
#
#       输入观测
#          │
#   ┌──────▼──────────────────────────────┐
#   │     FeatureExtractor (可替换)        │  ← 当前: MLPFeatureExtractor
#   │     （未来替换为 VisionExtractor）   │  ← 未来: VisionFeatureExtractor
#   └──────┬──────────────────────────────┘
#          │  feature_vector (feature_dim,)
#     ┌────┴────┐
#     ▼         ▼
#  ActorHead  CriticHead
#     │         │
#  μ, σ(连续)   V(s)
#
# 好处：未来引入视觉输入（RGB/Depth）时，
#       只需替换 FeatureExtractor，PPO 主体代码零修改。
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, List

from config import PPOConfig

# 1. 在 networks.py 顶部找个位置加入这个类（这是我的 PyTorch 归一化方案）
class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def update(self, x: torch.Tensor):
        with torch.no_grad():
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]

            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
            new_var = m2 / tot_count

            self.mean = new_mean
            self.var = new_var
            self.count = tot_count

    def forward(self, x: torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


# =============================================================================
# 特征提取器基类（定义接口规范）
# =============================================================================

class BaseFeatureExtractor(nn.Module):
    """
    特征提取器基类。
    所有特征提取器必须继承此类，并实现 forward 方法。
    输出维度固定为 feature_dim，以保证 Actor/Critic 头部不需要修改。
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入观测张量，具体 shape 由子类决定
        Returns:
            features: shape=(batch_size, feature_dim) 的特征向量
        """
        raise NotImplementedError


# =============================================================================
# 当前版本：MLP 特征提取器（处理 1D 向量观测）
# =============================================================================

class MLPFeatureExtractor(BaseFeatureExtractor):
    """
    多层感知机特征提取器。
    用于处理 1D 向量输入（6维基础观测 + 射线感知展开向量）。

    网络结构:
        Linear(state_dim → hidden[0]) → LayerNorm → ELU
        Linear(hidden[0] → hidden[1]) → LayerNorm → ELU
        ...
        Linear(hidden[-1] → feature_dim) → ELU

    为何用 LayerNorm 而非 BatchNorm:
        强化学习 batch size 小且样本相关性强，BatchNorm 不稳定。
        LayerNorm 在每个样本内做归一化，不受 batch size 影响。

    为何用 ELU 而非 ReLU:
        ELU 在负区间有非零梯度，缓解 "dying neuron" 问题，
        对连续控制任务（输入可能有负值）更友好。
    """

    def __init__(self, state_dim: int, hidden_dims: List[int], feature_dim: int):
        """
        Args:
            state_dim   : 输入向量维度（从 config.state_dim 读取）
            hidden_dims : 隐藏层维度列表，如 [256, 256]
            feature_dim : 输出特征维度，对应 config.feature_dim
        """
        super().__init__(feature_dim)

        # 动态构建多层 MLP
        # dims 形如 [state_dim, 256, 256, feature_dim]
        dims = [state_dim] + hidden_dims + [feature_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ELU())

        self.net = nn.Sequential(*layers)

        # 使用正交初始化：对 RL 的 MLP 来说比默认的 Xavier 更稳定
        self._init_weights()

    def _init_weights(self):
        """正交初始化权重（推荐用于 PPO，见 Andrychowicz et al. 2021）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape=(batch_size, state_dim) 的向量观测
        Returns:
            features: shape=(batch_size, feature_dim)
        """
        return self.net(x)


# =============================================================================
# 预留接口：视觉特征提取器（未来 Sim2Real 接入 CNN 时在此实现）
# =============================================================================

class VisionFeatureExtractor(BaseFeatureExtractor):
    """
    【预留接口 - 当前未实现，Sim2Real 阶段在此扩展】

    用途：处理来自真实摄像头的 2D 图像输入（RGB / Depth / 语义分割）。
    当接入此提取器时，只需将 UGVPPOAgent 初始化时的 feature_extractor 参数
    替换为此类的实例，PPO 主体代码（ppo_algorithm.py）无需任何修改。

    未来实现建议：
        输入: image tensor, shape=(batch, C, H, W)
            - RGB:   C=3
            - Depth: C=1
            - RGBD:  C=4（拼接）

        网络骨干选项（按复杂度排序）：
            A. 轻量 CNN (推荐 Sim2Real 起步)
               Conv2d(C→32, 8×8, stride=4) → ReLU
               Conv2d(32→64, 4×4, stride=2) → ReLU
               Conv2d(64→64, 3×3, stride=1) → ReLU → Flatten → Linear → feature_dim

            B. Nature DQN CNN（经典基线）

            C. ResNet18 (pretrained, 适合真实图像的 Sim2Real Fine-tuning)

        与向量观测融合（多模态）：
            # 示例：图像特征 + 速度/姿态向量特征 → concat → Linear → feature_dim
            image_feat = self.cnn(image)  # (B, cnn_out_dim)
            vec_feat   = self.vec_net(vec_obs)  # (B, vec_out_dim)
            fused      = torch.cat([image_feat, vec_feat], dim=-1)  # (B, cnn_out_dim + vec_out_dim)
            features   = self.fusion_layer(fused)  # (B, feature_dim)

    TODO: 在进行 Sim2Real 迁移实验时，实现此类，并在 train.py 中切换。
    """

    def __init__(self, image_channels: int, image_height: int, image_width: int, feature_dim: int):
        super().__init__(feature_dim)
        # TODO: 在此定义 CNN 层
        # self.cnn = nn.Sequential(...)
        # self.flatten_fc = nn.Linear(cnn_flat_dim, feature_dim)
        raise NotImplementedError("VisionFeatureExtractor 尚未实现，Sim2Real 阶段请在此扩展！")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 实现前向传播
        raise NotImplementedError


# =============================================================================
# Actor 网络：输出动作分布的均值和标准差
# =============================================================================

class Actor(nn.Module):
    """
    策略网络 (Policy Network / Actor)。

    输出连续动作的高斯分布参数：
        μ (mu): 动作均值，通过 tanh 映射到 [-1, 1]，对应 [-1,1] 的 Unity 动作空间
        σ (std): 动作标准差，以可学习参数形式存在（与状态无关，是全局参数）

    为何 std 不依赖状态：
        State-independent std 在连续控制 PPO 中是标准做法（OpenAI PPO 原始实现），
        更稳定且足够灵活。State-dependent std 虽然理论上更强，但训练难度更高。
    """

    def __init__(self, feature_extractor: BaseFeatureExtractor, action_dim: int):
        """
        Args:
            feature_extractor: 可插拔的特征提取器（MLP 或 Vision）
            action_dim: 动作维度，此处为 2 (Steer, Motor)
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        feature_dim = feature_extractor.feature_dim

        # 输出均值的线性层（均值头部）
        # 初始化为小增益，确保策略初始化时接近均匀分布
        self.mu_head = nn.Linear(feature_dim, action_dim)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)

        # 可学习的对数标准差（log_std）参数
        # 使用 log_std 而非 std 是为了保证 std > 0（exp 的值域为正数）
        # 初始化为 0 意味着初始 std = exp(0) = 1.0，动作探索充分
        # [O-3 修复] 初始 std 从 1.0 降为 ~0.6，减少无效的边界采样
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回动作分布的均值和标准差。

        Args:
            obs: shape=(batch_size, state_dim)
        Returns:
            mu : shape=(batch_size, action_dim)  动作均值，映射到 [-1,1]
            std: shape=(action_dim,) 或 (batch_size, action_dim)  动作标准差
        """
        # 步骤1: 提取特征  (batch, state_dim) → (batch, feature_dim)
        features = self.feature_extractor(obs)

        # 步骤2: 计算动作均值，用 tanh 压缩到 [-1, 1]
        # tanh 保证 Unity 动作空间约束（Unity 的连续动作默认在 [-1,1]）
        # 去掉 tanh! 直接输出! mu = torch.tanh(self.mu_head(features))  # (batch, action_dim)

        # 步骤3: 计算标准差
        # log_std 是可学习参数，将其 exp 后得到正数的 std
        # clamp 防止 std 过大或过小导致数值不稳定
        mu = self.mu_head(features)  
        std = torch.exp(self.log_std.clamp(-20, 2))  # (action_dim,)
        # expand_as 将 std 广播到与 mu 相同的 shape，便于后续构建分布
        std = std.expand_as(mu)  # (batch, action_dim)

        return mu, std

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作，同时返回 log_prob（采集数据时用）。

        Returns:
            action  : shape=(batch_size, action_dim)  采样得到的动作
            log_prob: shape=(batch_size,)              动作对数概率（用于 PPO ratio）
            entropy : shape=(batch_size,)              分布熵（用于 Entropy Bonus）
        """
        mu, std = self.forward(obs)

        # 构建独立高斯分布（每个动作维度独立）
        dist = Normal(mu, std)

        # 从分布中采样原始动作（未截断）
        raw_action = dist.sample()  # (batch, action_dim)

        # [M-1 修复] 先用未截断的原始动作计算 log_prob，保证概率计算准确
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # 再截断到 [-1, 1]，发给 Unity 执行
        # action = raw_action.clamp(-1.0, 1.0)

        # 计算熵：sum over action dimensions，再求均值（标量）
        entropy = dist.entropy().sum(dim=-1)  # (batch,)

        return raw_action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对已有动作（采集时存储的）重新计算 log_prob 和 entropy（更新策略时用）。

        Args:
            obs    : shape=(batch_size, state_dim)
            actions: shape=(batch_size, action_dim)  采集时记录的动作
        Returns:
            log_prob: shape=(batch_size,)
            entropy : shape=(batch_size,)
        """
        mu, std = self.forward(obs)
        dist = Normal(mu, std)

        # 注意：这里用传入的 actions（采集时的旧动作），而非重新采样
        # 这是 PPO 计算 importance ratio 的关键：r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


# =============================================================================
# Critic 网络：输出状态价值 V(s)
# =============================================================================

class Critic(nn.Module):
    """
    价值网络 (Value Network / Critic)。

    输出状态价值 V(s)，即在状态 s 下遵循当前策略能获得的期望累积回报。
    Critic 的准确性直接影响 GAE 优势函数的质量，进而影响 Actor 更新的方差。

    注意：Actor 和 Critic 拥有各自独立的特征提取器（不共享参数）。
    虽然共享特征提取器能节省参数，但对于 PPO 来说，独立提取器训练更稳定，
    且方便未来单独修改 Actor/Critic 网络容量（如：Asymmetric Actor-Critic）。
    """

    def __init__(self, feature_extractor: BaseFeatureExtractor):
        super().__init__()

        self.feature_extractor = feature_extractor
        feature_dim = feature_extractor.feature_dim

        # 价值头：输出单个标量 V(s)
        self.value_head = nn.Linear(feature_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: shape=(batch_size, state_dim)
        Returns:
            value: shape=(batch_size, 1)  状态价值估计
        """
        features = self.feature_extractor(obs)  # (batch, feature_dim)
        value = self.value_head(features)         # (batch, 1)
        return value