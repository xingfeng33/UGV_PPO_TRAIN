# =============================================================================
# 模块 5: 日志与曲线分析 (Logging & TensorBoard)
#
# 记录的指标（直接对应论文实验图表）：
#   - Train/EpisodeReward    : 训练时每 episode 的累积奖励（主要学习曲线）
#   - Train/EpisodeLength    : 每 episode 的步数（反映探索效率）
#   - Train/ActorLoss        : Actor 损失（监控策略更新稳定性）
#   - Train/CriticLoss       : Critic 损失（监控价值估计准确性）
#   - Train/Entropy          : 策略熵（监控探索程度，不应过快下降）
#   - Train/LearningRate     : 学习率（如使用 LR 调度时）
#   - Eval/MeanReward        : 评估时的平均奖励（论文主要对比指标）
#   - Eval/SuccessRate        : 成功率（到达目标的 episode 比例）
# =============================================================================

import os
import time
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


class TrainingLogger:
    """
    统一的训练日志管理器。
    同时输出到：
      1. TensorBoard（用于论文画图）
      2. 控制台（用于训练时实时监控）
    """

    def __init__(self, log_dir: str, run_name: Optional[str] = None):
        """
        Args:
            log_dir : TensorBoard 日志根目录（config.log_dir）
            run_name: 本次运行的标识名，默认使用时间戳
                      建议命名规则: "exp1_baseline_seed42"
                      便于 TensorBoard 中区分不同实验
        """
        if run_name is None:
            run_name = time.strftime("%Y%m%d_%H%M%S")

        self.full_log_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_dir=self.full_log_dir)

        # 滑动窗口统计（近 100 episode 的平均，更平滑）
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        # 当前 episode 的累积量
        self._current_episode_reward = 0.0
        self._current_episode_length = 0

        # 训练开始时间（计算 FPS 用）
        self._start_time = time.time()
        self._last_log_step = 0

        print(f"[Logger] TensorBoard 日志目录: {self.full_log_dir}")
        print(f"[Logger] 启动 TensorBoard: tensorboard --logdir {log_dir}")

    def step(self, reward: float, done: bool):
        """
        每个环境步调用一次，累积 episode 统计信息。

        Args:
            reward: 当前步即时奖励
            done  : 当前步是否结束
        """
        self._current_episode_reward += reward
        self._current_episode_length += 1

        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_lengths.append(self._current_episode_length)
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

    def log_training(self, global_step: int, actor_loss: float,
                     critic_loss: float, entropy: float, lr: float):
        """
        每次策略更新后调用，记录算法损失指标到 TensorBoard。

        Args:
            global_step : 全局时间步数（TensorBoard 横轴）
            actor_loss  : Actor 损失均值
            critic_loss : Critic 损失均值
            entropy     : 策略熵均值（越低说明策略越确定）
            lr          : 当前学习率
        """
        self.writer.add_scalar("Train/ActorLoss",  actor_loss,  global_step)
        self.writer.add_scalar("Train/CriticLoss", critic_loss, global_step)
        self.writer.add_scalar("Train/Entropy",    entropy,     global_step)
        self.writer.add_scalar("Train/LearningRate", lr,        global_step)

        # 记录 episode 滑动平均奖励
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths)
            self.writer.add_scalar("Train/EpisodeReward_Mean100", mean_reward, global_step)
            self.writer.add_scalar("Train/EpisodeLength_Mean100", mean_length, global_step)

        # 计算并记录训练吞吐量（FPS）
        elapsed = time.time() - self._start_time
        fps = (global_step - self._last_log_step) / max(elapsed, 1e-8)
        self.writer.add_scalar("Train/FPS", fps, global_step)
        self._last_log_step = global_step
        self._start_time = time.time()  # reset for next interval

        # 控制台输出（每次更新都打印，方便监控）
        mean_r = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        print(
            f"[Step {global_step:>8d}] "
            f"EpReward(100): {mean_r:>8.2f} | "
            f"ActorLoss: {actor_loss:>8.4f} | "
            f"CriticLoss: {critic_loss:>8.4f} | "
            f"Entropy: {entropy:>6.4f} | "
            f"FPS: {fps:>6.0f}"
        )

    def log_episode(self, global_step: int, episode_reward: float, episode_length: int):
        """
        每个 episode 结束时调用，直接记录单 episode 数据点（不做滑动平均）。
        适合绘制原始学习曲线（散点图）。
        """
        self.writer.add_scalar("Train/EpisodeReward", episode_reward, global_step)
        self.writer.add_scalar("Train/EpisodeLength", episode_length, global_step)

    def log_eval(self, global_step: int, mean_reward: float,
                 std_reward: float, success_rate: float):
        """
        评估结束后调用，记录评估指标。

        Args:
            mean_reward : 评估 episodes 的平均奖励
            std_reward  : 奖励标准差（反映策略稳定性，论文中常用 mean±std 表示）
            success_rate: 成功到达目标的比例（论文中的主要性能指标）
        """
        self.writer.add_scalar("Eval/MeanReward",   mean_reward,  global_step)
        self.writer.add_scalar("Eval/StdReward",    std_reward,   global_step)
        self.writer.add_scalar("Eval/SuccessRate",  success_rate, global_step)

        print(
            f"\n{'='*60}\n"
            f"[Eval @ Step {global_step}] "
            f"MeanReward: {mean_reward:.2f} ± {std_reward:.2f} | "
            f"SuccessRate: {success_rate*100:.1f}%\n"
            f"{'='*60}\n"
        )

    def close(self):
        """训练结束后调用，确保 TensorBoard 数据刷盘"""
        self.writer.flush()
        self.writer.close()
        print("[Logger] TensorBoard writer 已关闭，数据已保存。")