# =============================================================================
# 模块 4: PPO 算法核心 (PPO Algorithm)
#
# 包含三大组件：
#   1. RolloutBuffer: 存储一轮采集的轨迹数据，计算 GAE 优势函数
#   2. PPOAgent: 封装 Actor + Critic，提供 select_action / update 接口
#   3. PPOTrainer: 管理整体训练流程的调度
#
# 关键公式（将在代码注释中逐一标注）：
#   GAE: A_t = Σ (γλ)^k * δ_{t+k}，其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
#   Actor Loss: L_CLIP = -E[min(r_t*A_t, clip(r_t, 1-ε, 1+ε)*A_t)]
#   Critic Loss: L_VF = E[(V_θ(s_t) - V_target_t)^2]
#   Total Loss: L = L_CLIP + c1*L_VF - c2*H[π]
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Generator

from config import PPOConfig
from networks import Actor, Critic, MLPFeatureExtractor, RunningMeanStd


# =============================================================================
# Rollout Buffer：轨迹数据存储与 GAE 计算
# =============================================================================

class RolloutBuffer:
    """
    存储一次 rollout 采集的 (s, a, r, s', done) 序列数据。

    核心功能：
    1. 存储：在与环境交互过程中逐步填充数据
    2. 计算 GAE：一次 rollout 结束后，批量计算 GAE 优势函数
    3. 采样：按 mini-batch 大小随机打乱并输出，供策略更新使用

    内存估算：
        假设 rollout_steps=4096，state_dim=54，action_dim=2
        → 约需 4096 * (54 + 2 + 1 + 1 + 1 + 1) * 4 bytes ≈ 2.4 MB，非常轻量
    """

    def __init__(self, config: PPOConfig, state_dim: int, action_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 分配缓冲区（numpy 存储，GPU 计算时再转 tensor）
        n = config.rollout_steps
        self.states      = np.zeros((n, state_dim),   dtype=np.float32)
        self.actions     = np.zeros((n, action_dim),  dtype=np.float32)
        self.rewards     = np.zeros(n,                dtype=np.float32)
        self.dones       = np.zeros(n,                dtype=np.float32)  # 1.0 = done
        self.log_probs   = np.zeros(n,                dtype=np.float32)  # 采集时的旧 log_prob
        self.values      = np.zeros(n,                dtype=np.float32)  # Critic 估计的 V(s_t)

        # GAE 计算完成后填充
        self.advantages  = np.zeros(n,                dtype=np.float32)
        self.returns     = np.zeros(n,                dtype=np.float32)  # V_target = advantage + value

        self.ptr = 0          # 当前写入位置指针
        self.full = False     # 缓冲区是否已满

    def add(self, state, action, reward, done, log_prob, value):
        """向缓冲区追加一步数据"""
        assert self.ptr < self.config.rollout_steps, "缓冲区已满，请先调用 compute_gae() 再 reset()"

        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr += 1

    def compute_gae(self, last_value: float):
        """
        计算 GAE (Generalized Advantage Estimation) 优势函数。

        公式（Schulman et al. 2015b）：
            δ_t  = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
                   ↑TD误差：即时奖励+下一状态价值-当前状态价值

            A_t  = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
                 = δ_t + (γλ) * A_{t+1} * (1 - done_t)    ← 递推形式，从后往前计算

        参数说明：
            last_value: 最后一步之后的 V(s_{T+1})
                        如果最后一步 done=True，则为 0
                        如果最后一步 done=False（时间截断），则为 Critic 对下一状态的估计

        数学意义：
            - γ（gamma）：折扣因子，控制对未来奖励的重视程度
            - λ（gae_lambda）：GAE 中 bias-variance tradeoff 参数
              λ→0: 高偏差低方差（类似 TD(0)）
              λ→1: 低偏差高方差（类似 MC）
              λ=0.95 是连续控制任务的经典设置
        """
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        n = self.ptr  # 实际填充的步数（可能 < rollout_steps，最后一个 episode 截断时）

        gae = 0.0  # 从末尾开始累积的 GAE 值

        for t in reversed(range(n)):
            # 下一时刻的价值估计：如果是 done 则下一状态价值为 0
            if t == n - 1:
                # 最后一步：使用传入的 last_value 作为 bootstrap
                next_value = last_value * (1.0 - self.dones[t])
            else:
                # 中间步：使用下一步的 Critic 估计值
                next_value = self.values[t + 1] * (1.0 - self.dones[t])

            # TD 误差 δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + gamma * next_value - self.values[t]

            # GAE 递推：A_t = δ_t + γλ * A_{t+1} * (1 - done_t)
            # done 处截断 GAE（不跨越 episode 边界累积）
            gae = delta + gamma * gae_lambda * gae * (1.0 - self.dones[t])

            self.advantages[t] = gae

        # V_target（Critic 训练目标）= 优势 + 基线价值
        # 即 V_target_t = A_t + V(s_t)（这就是 TD(λ) 回报的近似）
        self.returns[:n] = self.advantages[:n] + self.values[:n]

        # 归一化优势函数（Advantage Normalization）
        # 原因：不同 episode 奖励尺度不同，归一化后梯度更稳定
        # 论文参考：这是 PPO 实现中的重要 trick（Engstrom et al. 2020 "Implementation Matters"）
        adv = self.advantages[:n]
        self.advantages[:n] = (adv - adv.mean()) / (adv.std() + 1e-8)

    def get_mini_batches(self) -> Generator:
        """
        将 rollout buffer 随机打乱后，按 mini_batch_size 切分，逐批 yield。

        为何要打乱：PPO 的更新假设数据独立同分布，打乱破坏时序相关性。
        """
        n = self.ptr
        batch_size = self.config.mini_batch_size

        # 随机打乱索引
        indices = np.random.permutation(n)

        # 转换为 GPU tensor（一次性转移，减少 CPU-GPU 通信次数）
        states_t   = torch.from_numpy(self.states[:n]).to(self.device)
        actions_t  = torch.from_numpy(self.actions[:n]).to(self.device)
        log_probs_t = torch.from_numpy(self.log_probs[:n]).to(self.device)
        advantages_t = torch.from_numpy(self.advantages[:n]).to(self.device)
        returns_t  = torch.from_numpy(self.returns[:n]).to(self.device)

        # 按 mini_batch_size 切分并 yield
        start = 0
        while start < n:
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield (
                states_t[batch_idx],       # (B, state_dim)
                actions_t[batch_idx],      # (B, action_dim)
                log_probs_t[batch_idx],    # (B,)
                advantages_t[batch_idx],   # (B,)
                returns_t[batch_idx],      # (B,)
            )
            start = end

    def reset(self):
        """清空缓冲区，准备下一次 rollout"""
        self.ptr = 0
        self.full = False


# =============================================================================
# PPO Agent：封装 Actor + Critic，对外提供干净接口
# =============================================================================

class PPOAgent:
    """
    PPO 智能体，封装：
      - Actor  (策略网络)
      - Critic (价值网络)
      - 优化器
      - 策略更新逻辑（含 PPO Clip Loss + Entropy Bonus + Critic Loss）
    """

    def __init__(self, config: PPOConfig, state_dim: int):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.obs_rms = RunningMeanStd(shape=(state_dim,)).to(self.device)

        print(f"[Agent] 使用计算设备: {self.device}")

        # ------------------------------------------------------------------
        # 构建 Actor 和 Critic（各自拥有独立的特征提取器）
        # 如未来要引入视觉，在此处将 MLPFeatureExtractor 替换为 VisionFeatureExtractor
        # ------------------------------------------------------------------
        actor_extractor = MLPFeatureExtractor(
            state_dim=state_dim,
            hidden_dims=config.mlp_hidden_dims,
            feature_dim=config.feature_dim
        )
        critic_extractor = MLPFeatureExtractor(
            state_dim=state_dim,
            hidden_dims=config.mlp_hidden_dims,
            feature_dim=config.feature_dim
        )

        self.actor  = Actor(actor_extractor,  config.action_dim).to(self.device)
        self.critic = Critic(critic_extractor).to(self.device)

        # 使用 Adam 优化器，同时优化 actor 和 critic 的所有参数
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.lr,
            eps=1e-5  # Adam epsilon，稳定性 trick（原 PPO 代码沿用）
        )

        # 全局步数计数器（用于 TensorBoard 横轴）
        self.global_step = 0
        # 回合计数器
        self.episode_count = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        根据当前观测选择动作（采集数据时调用，带梯度追踪关闭）。

        Args:
            obs: shape=(state_dim,) 的 numpy 数组
        Returns:
            action  : shape=(action_dim,) numpy 数组
            log_prob: float  动作对数概率（存入 buffer）
            value   : float  Critic 估计的状态价值（存入 buffer）
        """
        # numpy → tensor，增加 batch 维度：(state_dim,) → (1, state_dim)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()
        # 训练模式下更新均值方差
        if self.actor.training:
            self.obs_rms.update(obs_t)
        # 状态归一化
        obs_norm = self.obs_rms(obs_t)

        # Actor 采样动作
        # 拿到原始动作 raw_action
        raw_action_t, log_prob_t, _ = self.actor.get_action(obs_norm)

        # Critic 估计价值
        value_t = self.critic(obs_norm)  # (1, 1)

        # tensor → numpy，去除 batch 维度
        raw_action = raw_action_t.squeeze(0).cpu().numpy()
        log_prob = log_prob_t.squeeze(0).cpu().item()
        value    = value_t.squeeze().cpu().item()

        return raw_action, log_prob, value

    def update(self, buffer: RolloutBuffer) -> Tuple[float, float, float]:
        """
        用 rollout buffer 中的数据更新 Actor 和 Critic（PPO 的核心步骤）。

        Args:
            buffer: 已填充数据并计算完 GAE 的 RolloutBuffer

        Returns:
            (mean_actor_loss, mean_critic_loss, mean_entropy) 用于日志记录
        """
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0
        update_count      = 0

        # PPO 的精髓：对同一批数据重复更新 ppo_epochs 次
        for epoch in range(self.config.ppo_epochs):
            for batch in buffer.get_mini_batches():
                states, actions, old_log_probs, advantages, returns = batch
                # [核心新增]：更新网络时，把 Buffer 里的原始状态进行归一化
                states_norm = self.obs_rms(states)

                # ==========================================================
                # 步骤 1: 重新评估动作（用当前策略 π_θ 对旧动作 a 计算 log_prob）
                # ==========================================================
                # 接下来传入 states_norm，且 actions 是没有被截断过的原始动作！
                new_log_probs, entropy = self.actor.evaluate_actions(states_norm, actions)
                # new_log_probs: (B,)  当前策略对旧动作的 log 概率
                # entropy:       (B,)  当前策略的熵

                # ==========================================================
                # 步骤 2: 计算 Importance Sampling Ratio
                #
                #   r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
                #          = exp(log π_θ - log π_θ_old)
                #          = exp(new_log_prob - old_log_prob)
                #
                # 意义：衡量当前策略和采集数据时的策略差异程度
                # ==========================================================
                ratio = torch.exp(new_log_probs - old_log_probs)  # (B,)

                # ==========================================================
                # 步骤 3: 计算 PPO Clipped Actor Loss
                #
                #   L_CLIP = -E[ min(
                #               r_t(θ) * A_t,                              ← 未裁剪项
                #               clip(r_t(θ), 1-ε, 1+ε) * A_t              ← 裁剪项
                #            )]
                #
                # 含义：
                #   - 当 A_t > 0（好动作）：鼓励增大该动作概率，但不能增大超过 1+ε 倍
                #   - 当 A_t < 0（坏动作）：鼓励减小该动作概率，但不能减小超过 1-ε 倍
                # 这个约束防止策略更新步幅过大，是 PPO 稳定性的核心来源
                # ==========================================================
                surr1 = ratio * advantages                                          # (B,) 未裁剪
                surr2 = ratio.clamp(1.0 - self.config.clip_ratio,
                                    1.0 + self.config.clip_ratio) * advantages      # (B,) 裁剪后
                actor_loss = -torch.min(surr1, surr2).mean()
                # 注意取负号：PyTorch 做梯度下降（minimize），但我们想 maximize 目标函数

                # ==========================================================
                # 步骤 4: 计算 Critic Loss（均方误差）
                #
                #   L_VF = E[ (V_θ(s_t) - V_target_t)^2 ]
                #
                # V_target_t = GAE 优势 + 旧价值估计（即 buffer.returns）
                # 含义：让 Critic 学会准确预测期望回报
                # ==========================================================
                values_pred = self.critic(states).squeeze(-1)  # (B, 1) → (B,)
                critic_loss = F.mse_loss(values_pred, returns)
                # 使用 F.mse_loss 而非手写，数值等价但更简洁

                # ==========================================================
                # 步骤 5: 熵奖励（Entropy Bonus）
                #
                #   L_H = E[ H[π_θ(·|s_t)] ]
                #       = -E[ log π_θ(a_t|s_t) ]  （高斯分布的熵）
                #
                # 含义：鼓励策略保持随机性，防止过早收敛到确定性策略（探索正则化）
                # 在连续动作空间中尤其重要，防止 std 过快坍塌到 0
                # ==========================================================
                entropy_bonus = entropy.mean()

                # ==========================================================
                # 步骤 6: 组合总损失
                #
                #   L_total = L_CLIP + c1 * L_VF - c2 * L_H
                #
                # c1 = value_loss_coef（通常 0.5）：控制 Critic 更新速度
                # c2 = entropy_coef  （通常 0.01）：控制探索程度
                # 减号：因为我们想最大化熵，但 PyTorch 做最小化
                # ==========================================================
                total_loss = (actor_loss
                              + self.config.value_loss_coef * critic_loss
                              - self.config.entropy_coef * entropy_bonus)

                # ==========================================================
                # 步骤 7: 反向传播与梯度更新
                # ==========================================================
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪：防止梯度爆炸，这是 PPO 稳定训练的重要 trick
                # max_grad_norm 通常设 0.5，对 RL 任务尤其重要
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                # 累积损失用于日志
                total_actor_loss  += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy     += entropy_bonus.item()
                update_count      += 1

        mean_actor_loss  = total_actor_loss  / update_count
        mean_critic_loss = total_critic_loss / update_count
        mean_entropy     = total_entropy     / update_count

        return mean_actor_loss, mean_critic_loss, mean_entropy


# 在 ppo_algorithm.py 顶部补充此 import（Python 标准做法）
import torch.nn.functional as F